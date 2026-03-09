"""ManiSkill 3.0 Trajectory Ingestion Pipeline.

Collects robotics trajectories from ManiSkill GPU-parallelized environments,
compresses observations into latents via Cosmos-VAE, and stores them in HDF5
for efficient sliding-window training retrieval.

Usage:
    python data/ingest.py --task PickCube-v1 --num_envs 16 --max_episodes 10
"""

from __future__ import annotations

import argparse
import signal
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import gymnasium as gym
import h5py
import numpy as np
import torch
import torch.cuda.nvtx as nvtx


# ---------------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------------
@dataclass
class IngestConfig:
    task: str = "PickCube-v1"
    num_envs: int = 16
    max_episodes: int = 100
    max_steps: int = 200
    hdf5_path: str = "trajectories.h5"
    cosmos_ckpt: str = "nvidia/Cosmos-0.1-Tokenizer-CI16x16"
    seed: int = 42
    flush_every: int = 10  # flush HDF5 every N completed episodes

    @property
    def max_total_steps(self) -> int:
        return self.max_episodes * self.max_steps


# ---------------------------------------------------------------------------
# 2. ManiSkill Collector
# ---------------------------------------------------------------------------
class ManiSkillCollector:
    """Wraps a ManiSkill GPU-vectorized environment."""

    def __init__(self, cfg: IngestConfig):
        self.cfg = cfg
        self.env: Optional[gym.Env] = None

    def __enter__(self):
        self.env = gym.make(
            self.cfg.task,
            num_envs=self.cfg.num_envs,
            obs_mode="rgb",
            render_mode="sensors",
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.env is not None:
            self.env.close()
            self.env = None
        return False

    def reset(self):
        return self.env.reset(seed=self.cfg.seed)

    def sample_random_actions(self) -> torch.Tensor:
        """Sample random actions as a GPU tensor."""
        action_space = self.env.single_action_space
        actions = torch.rand(
            self.cfg.num_envs, *action_space.shape,
            dtype=torch.float32, device="cuda",
        )
        lo = torch.as_tensor(action_space.low, dtype=torch.float32, device="cuda")
        hi = torch.as_tensor(action_space.high, dtype=torch.float32, device="cuda")
        actions.mul_(hi - lo).add_(lo)
        return actions

    def step(self, actions: torch.Tensor):
        return self.env.step(actions)

    @property
    def action_dim(self) -> int:
        return int(np.prod(self.env.single_action_space.shape))


# ---------------------------------------------------------------------------
# 3. Cosmos Latent Encoder
# ---------------------------------------------------------------------------
class CosmosLatentEncoder:
    """Loads Cosmos-Tokenizer-CI16x16 and encodes RGB frames to latents."""

    def __init__(self, model_name: str = "nvidia/Cosmos-0.1-Tokenizer-CI16x16"):
        from cosmos_tokenizer.image_lib import ImageTokenizer  # type: ignore[import]

        self.encoder = ImageTokenizer(
            checkpoint_enc=model_name,
            device="cuda",
            dtype="float16",
        )
        self._halved = False

    @torch.no_grad()
    def encode(self, frame_buf: torch.Tensor) -> torch.Tensor:
        """Encode [B,3,H,W] float16 → [B,32,8,8] float16 latents.

        On OOM, retries with halved batch size and concatenates results.
        """
        nvtx.range_push("vae_encode")
        try:
            latents = self.encoder.encode(frame_buf)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            mid = frame_buf.shape[0] // 2
            l1 = self.encoder.encode(frame_buf[:mid])
            l2 = self.encoder.encode(frame_buf[mid:])
            latents = torch.cat([l1, l2], dim=0)
            self._halved = True
        nvtx.range_pop()
        return latents


# ---------------------------------------------------------------------------
# 4. HDF5 Writer
# ---------------------------------------------------------------------------
class HDF5Writer:
    """Manages per-episode HDF5 groups with buffered, chunked writes."""

    CHUNK_T = 16  # chunk size along time axis

    def __init__(self, path: str, action_dim: int, metadata: dict):
        self.path = path
        self.action_dim = action_dim
        self.metadata = metadata
        self.h5file: Optional[h5py.File] = None
        self._episode_count = 0
        self._buffers: dict[int, dict[str, list]] = {}  # env_id → field → list

    def __enter__(self):
        self.h5file = h5py.File(self.path, "w")
        meta_grp = self.h5file.create_group("metadata")
        for k, v in self.metadata.items():
            meta_grp.attrs[k] = v
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.h5file is not None:
            self._finalize_all()
            self.h5file.close()
            self.h5file = None
        return False

    def ensure_episode(self, env_id: int):
        """Start buffering a new episode for env_id if needed."""
        if env_id not in self._buffers:
            self._buffers[env_id] = {
                "latents": [],
                "actions": [],
                "rewards": [],
                "dones": [],
            }

    def append_frame(
        self,
        env_id: int,
        latent: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
    ):
        """Append a single frame for one environment."""
        buf = self._buffers[env_id]
        buf["latents"].append(latent)
        buf["actions"].append(action)
        buf["rewards"].append(reward)
        buf["dones"].append(done)

    def append_batch(
        self,
        latents: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        num_envs: int,
    ):
        """Append one step for all environments in the batch."""
        nvtx.range_push("hdf5_write")
        for i in range(num_envs):
            self.ensure_episode(i)
            self.append_frame(
                env_id=i,
                latent=latents[i],
                action=actions[i],
                reward=float(rewards[i]),
                done=bool(dones[i]),
            )
        nvtx.range_pop()

    def finalize_episode(self, env_id: int, task: str, seed: int, completed: bool):
        """Write buffered episode to HDF5 and clear buffer."""
        buf = self._buffers.pop(env_id, None)
        if buf is None or len(buf["latents"]) == 0:
            return

        T = len(buf["latents"])
        grp_name = f"episode_{self._episode_count:04d}"
        grp = self.h5file.create_group(grp_name)

        lat_arr = np.stack(buf["latents"], axis=0).astype(np.float16)
        act_arr = np.stack(buf["actions"], axis=0).astype(np.float32)
        rew_arr = np.array(buf["rewards"], dtype=np.float32)
        done_arr = np.array(buf["dones"], dtype=bool)

        C = self.CHUNK_T
        grp.create_dataset(
            "latents", data=lat_arr,
            chunks=(min(C, T), *lat_arr.shape[1:]),
            compression="lzf",
        )
        grp.create_dataset(
            "actions", data=act_arr,
            chunks=(min(C, T), self.action_dim),
            compression="lzf",
        )
        grp.create_dataset(
            "rewards", data=rew_arr,
            chunks=(min(C, T),),
            compression="lzf",
        )
        grp.create_dataset(
            "dones", data=done_arr,
            chunks=(min(C, T),),
        )

        grp.attrs["length"] = T
        grp.attrs["completed"] = completed
        grp.attrs["task"] = task
        grp.attrs["seed"] = seed
        grp.attrs["timestamp"] = datetime.now(timezone.utc).isoformat()

        self._episode_count += 1

    def flush(self):
        if self.h5file is not None:
            self.h5file.flush()

    def _finalize_all(self):
        """Finalize any remaining in-progress episodes on shutdown."""
        for env_id in list(self._buffers.keys()):
            self.finalize_episode(env_id, task="unknown", seed=0, completed=False)
        self.flush()

    @property
    def episode_count(self) -> int:
        return self._episode_count


# ---------------------------------------------------------------------------
# 5. LPS Benchmark
# ---------------------------------------------------------------------------
class LPSBenchmark:
    """Per-stage timing with CUDA events and NVTX ranges."""

    CUDA_STAGES = ("env_step", "preprocess", "vae_encode", "cpu_transfer")
    CPU_STAGES = ("hdf5_write",)
    ALL_STAGES = CUDA_STAGES + CPU_STAGES

    def __init__(self):
        self._start_events: dict[str, torch.cuda.Event] = {}
        self._end_events: dict[str, torch.cuda.Event] = {}
        self._cpu_times: dict[str, list[float]] = {s: [] for s in self.CPU_STAGES}
        self._cuda_elapsed: dict[str, list[float]] = {s: [] for s in self.CUDA_STAGES}
        self._total_start: float = 0.0
        self._total_end: float = 0.0

        for stage in self.CUDA_STAGES:
            self._start_events[stage] = []
            self._end_events[stage] = []

    def start_total(self):
        torch.cuda.synchronize()
        self._total_start = time.perf_counter()

    def stop_total(self):
        torch.cuda.synchronize()
        self._total_end = time.perf_counter()

    def cuda_start(self, stage: str):
        """Record start CUDA event and push NVTX range."""
        nvtx.range_push(stage)
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
        self._start_events[stage].append(ev)

    def cuda_stop(self, stage: str):
        """Record end CUDA event and pop NVTX range."""
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
        self._end_events[stage].append(ev)
        nvtx.range_pop()

    def cpu_start(self, stage: str):
        nvtx.range_push(stage)
        self._cpu_times[stage].append(-time.perf_counter())

    def cpu_stop(self, stage: str):
        self._cpu_times[stage][-1] += time.perf_counter()
        nvtx.range_pop()

    def summarize(self, num_envs: int, num_steps: int):
        """Print LPS and per-stage breakdown. Called once at end."""
        torch.cuda.synchronize()
        wall = self._total_end - self._total_start
        lps = (num_envs * num_steps) / wall if wall > 0 else 0.0

        stage_ms: dict[str, float] = {}

        for stage in self.CUDA_STAGES:
            starts = self._start_events[stage]
            ends = self._end_events[stage]
            total_ms = sum(
                s.elapsed_time(e) for s, e in zip(starts, ends)
            )
            stage_ms[stage] = total_ms

        for stage in self.CPU_STAGES:
            total_ms = sum(self._cpu_times[stage]) * 1000.0
            stage_ms[stage] = total_ms

        total_staged = sum(stage_ms.values())

        print("\n" + "=" * 60)
        print(f"LPS Benchmark  —  {lps:,.0f} latents/sec  ({wall:.2f}s wall)")
        print("=" * 60)
        print(f"{'Stage':<16} {'Time (ms)':>12} {'% Total':>10}")
        print("-" * 40)
        for stage in self.ALL_STAGES:
            ms = stage_ms[stage]
            pct = 100.0 * ms / total_staged if total_staged > 0 else 0.0
            print(f"{stage:<16} {ms:>12.1f} {pct:>9.1f}%")
        print("-" * 40)
        print(f"{'TOTAL':<16} {total_staged:>12.1f}")
        print(f"{'Wall clock':<16} {wall * 1000:>12.1f}")
        print("=" * 60)


# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------
_shutdown_flag = False


def _signal_handler(signum, frame):
    global _shutdown_flag
    _shutdown_flag = True


# ---------------------------------------------------------------------------
# Main ingestion loop
# ---------------------------------------------------------------------------
def run(cfg: IngestConfig):
    global _shutdown_flag
    _shutdown_flag = False

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    torch.manual_seed(cfg.seed)

    # Pre-allocate frame buffer (zero-allocation in hot loop)
    frame_buf = torch.empty(
        cfg.num_envs, 3, 128, 128,
        dtype=torch.float16, device="cuda",
    )

    encoder = CosmosLatentEncoder(cfg.cosmos_ckpt)
    bench = LPSBenchmark()

    with ManiSkillCollector(cfg) as collector, \
         HDF5Writer(
             path=cfg.hdf5_path,
             action_dim=collector.action_dim,
             metadata={
                 "cosmos_model": cfg.cosmos_ckpt,
                 "compression_ratio": 256,  # 16x16 spatial
                 "obs_resolution": 128,
             },
         ) as writer:

        # Reset environments
        obs, _ = collector.reset()
        for i in range(cfg.num_envs):
            writer.ensure_episode(i)

        completed_episodes = 0
        step_count = 0
        bench.start_total()

        for step in range(cfg.max_total_steps):
            if _shutdown_flag:
                break
            if completed_episodes >= cfg.max_episodes:
                break

            # --- 1. Sample actions & step environment (GPU) ---
            bench.cuda_start("env_step")
            actions = collector.sample_random_actions()
            obs, rewards, terms, truncs, infos = collector.step(actions)
            bench.cuda_stop("env_step")

            # --- 2. Preprocess RGB → float16 NCHW (in-place, GPU) ---
            bench.cuda_start("preprocess")
            rgb = obs["sensor_data"]["base_camera"]["rgb"]  # [N,128,128,4] uint8
            frame_buf.copy_(
                rgb[:, :, :, :3].permute(0, 3, 1, 2).to(torch.float16)
            )
            frame_buf.div_(255.0)
            bench.cuda_stop("preprocess")

            # --- 3. VAE encode (GPU, float16, no_grad) ---
            bench.cuda_start("vae_encode")
            latents = encoder.encode(frame_buf)
            bench.cuda_stop("vae_encode")

            # --- 4. CPU transfer ---
            bench.cuda_start("cpu_transfer")
            latents_cpu = latents.cpu().numpy()
            actions_cpu = actions.cpu().numpy()
            rewards_cpu = rewards.cpu().numpy()
            dones_cpu = (terms | truncs).cpu().numpy()
            bench.cuda_stop("cpu_transfer")

            # --- 5. HDF5 write ---
            bench.cpu_start("hdf5_write")
            writer.append_batch(
                latents_cpu, actions_cpu, rewards_cpu, dones_cpu,
                num_envs=cfg.num_envs,
            )
            bench.cpu_stop("hdf5_write")

            step_count += 1

            # --- 6. Handle episode resets for done envs ---
            done_mask = terms | truncs  # GPU bool tensor
            done_indices = done_mask.nonzero(as_tuple=False).squeeze(-1)
            if done_indices.numel() > 0:
                done_list = done_indices.cpu().tolist()
                for env_id in done_list:
                    writer.finalize_episode(
                        env_id=env_id,
                        task=cfg.task,
                        seed=cfg.seed,
                        completed=bool(terms[env_id].item()),
                    )
                    completed_episodes += 1
                    writer.ensure_episode(env_id)

                    if completed_episodes % cfg.flush_every == 0:
                        writer.flush()

                    if completed_episodes >= cfg.max_episodes:
                        break

        bench.stop_total()
        bench.summarize(num_envs=cfg.num_envs, num_steps=step_count)

        print(f"\nCompleted {completed_episodes} episodes, "
              f"{writer.episode_count} written to {cfg.hdf5_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def parse_args() -> IngestConfig:
    p = argparse.ArgumentParser(description="ManiSkill → Cosmos-VAE → HDF5 pipeline")
    p.add_argument("--task", type=str, default="PickCube-v1")
    p.add_argument("--num_envs", type=int, default=16)
    p.add_argument("--max_episodes", type=int, default=100)
    p.add_argument("--max_steps", type=int, default=200)
    p.add_argument("--hdf5_path", type=str, default="trajectories.h5")
    p.add_argument("--cosmos_ckpt", type=str, default="nvidia/Cosmos-0.1-Tokenizer-CI16x16")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--flush_every", type=int, default=10)
    args = p.parse_args()
    return IngestConfig(**vars(args))


if __name__ == "__main__":
    cfg = parse_args()
    run(cfg)
