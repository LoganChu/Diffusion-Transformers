"""ManiSkill 3.0 Trajectory Ingestion Pipeline.

Collects robotics trajectories from ManiSkill GPU-parallelized environments,
compresses observations into latents via Cosmos-VAE, and stores them in HDF5
for efficient sliding-window training retrieval.

Usage:
    python data/ingest.py --task PickCube-v1 --num_envs 64 --max_episodes 10000

Optimizations (v2):
  - Double-buffered VAE on a separate CUDA stream (overlaps encode with env.step)
  - Async HDF5 writer on a dedicated process via multiprocessing.Queue
  - rdcc_nbytes=64MB chunk cache for sequential write performance
  - Increased default num_envs=64 to better amortize physics step cost

Note: SAPIEN's Vulkan-CUDA interop crashes on RTX 50-series Windows.
Until fixed upstream, use obs_mode="state" with random projection (default).
Pass --rgb_obs once the SAPIEN renderer is patched.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import signal
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import gymnasium as gym
import h5py
import mani_skill.envs  # noqa: F401 — register ManiSkill envs
import numpy as np
import torch
import torch.cuda.nvtx as nvtx

LATENT_C = 16
LATENT_H = 8
LATENT_W = 8
OBS_H = 128
OBS_W = 128


# ---------------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------------
@dataclass
class IngestConfig:
    task: str = "PickCube-v1"
    num_envs: int = 128
    max_episodes: int = 100
    max_steps: int = 200
    hdf5_path: str = "trajectories.h5"
    cosmos_ckpt: str = "pretrained_ckpts/Cosmos-Tokenizer-CI16x16"
    seed: int = 42
    flush_every: int = 50
    rdcc_nbytes: int = 64 * 1024 * 1024
    rgb_obs: bool = False
    async_writer: bool = True  # use multiprocess HDF5 writer

    @property
    def max_total_steps(self) -> int:
        return self.max_episodes * self.max_steps


# ---------------------------------------------------------------------------
# Weight download helper
# ---------------------------------------------------------------------------
def ensure_cosmos_weights(ckpt_dir: str) -> str:
    encoder_jit = os.path.join(ckpt_dir, "encoder.jit")
    if os.path.isfile(encoder_jit):
        return encoder_jit
    from huggingface_hub import snapshot_download
    os.makedirs(ckpt_dir, exist_ok=True)
    snapshot_download(repo_id="nvidia/Cosmos-0.1-Tokenizer-CI16x16", local_dir=ckpt_dir)
    assert os.path.isfile(encoder_jit), f"encoder.jit not found in {ckpt_dir}"
    return encoder_jit


# ---------------------------------------------------------------------------
# 2. ManiSkill Collector
# ---------------------------------------------------------------------------
class ManiSkillCollector:
    """Wraps ManiSkill GPU-vectorized env. Supports state-to-RGB projection."""

    def __init__(self, cfg: IngestConfig):
        self.cfg = cfg
        self.env: Optional[gym.Env] = None
        self._proj: Optional[torch.Tensor] = None

    def __enter__(self):
        obs_mode = "rgb" if self.cfg.rgb_obs else "state"
        kwargs = dict(num_envs=self.cfg.num_envs, obs_mode=obs_mode)
        if self.cfg.rgb_obs:
            kwargs["render_mode"] = "sensors"
        self.env = gym.make(self.cfg.task, **kwargs)

        if not self.cfg.rgb_obs:
            state_dim = self.env.unwrapped.single_observation_space.shape[0]
            gen = torch.Generator(device="cuda").manual_seed(self.cfg.seed + 9999)
            self._proj = torch.randn(
                state_dim, 3 * OBS_H * OBS_W,
                dtype=torch.float16, device="cuda", generator=gen,
            )
            self._proj.mul_(0.5 / (state_dim ** 0.5))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.env is not None:
            self.env.close()
            self.env = None
        return False

    def reset(self):
        return self.env.reset(seed=self.cfg.seed)

    def sample_random_actions(self) -> torch.Tensor:
        action_space = self.env.unwrapped.single_action_space
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

    def extract_rgb(self, obs, frame_buf: torch.Tensor):
        """Extract/synthesize RGB into pre-allocated frame_buf [N,3,H,W] float16."""
        if self.cfg.rgb_obs:
            rgb = obs["sensor_data"]["base_camera"]["rgb"]
            frame_buf.copy_(rgb[:, :, :, :3].permute(0, 3, 1, 2).to(torch.float16))
            frame_buf.div_(255.0)
        else:
            projected = obs.to(torch.float16) @ self._proj
            frame_buf.copy_(projected.sigmoid_().view(-1, 3, OBS_H, OBS_W))

    @property
    def action_dim(self) -> int:
        return int(np.prod(self.env.unwrapped.single_action_space.shape))


# ---------------------------------------------------------------------------
# 3. Cosmos Latent Encoder (with separate CUDA stream)
# ---------------------------------------------------------------------------
class CosmosLatentEncoder:
    """Cosmos-Tokenizer-CI16x16 encoder with optional CUDA stream pipelining.

    Input:  [B, 3, 128, 128] float16 in [0,1]
    Output: [B, 16, 8, 8]    float16
    """

    def __init__(self, ckpt_dir: str):
        encoder_jit = ensure_cosmos_weights(ckpt_dir)
        from cosmos_tokenizer.image_lib import ImageTokenizer
        self.encoder = ImageTokenizer(checkpoint_enc=encoder_jit)
        self.stream = torch.cuda.Stream()
        self._halved = False

    @torch.no_grad()
    def encode(self, frame_buf: torch.Tensor) -> torch.Tensor:
        """Encode on a separate CUDA stream for overlap with env.step."""
        with torch.cuda.stream(self.stream):
            bf16_buf = frame_buf.to(torch.bfloat16)
            try:
                (latents,) = self.encoder.encode(bf16_buf)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                mid = bf16_buf.shape[0] // 2
                (l1,) = self.encoder.encode(bf16_buf[:mid])
                (l2,) = self.encoder.encode(bf16_buf[mid:])
                latents = torch.cat([l1, l2], dim=0)
                self._halved = True
        return latents.to(torch.float16)

    def sync(self):
        """Wait for the encode stream to finish."""
        self.stream.synchronize()


# ---------------------------------------------------------------------------
# 4. HDF5 Writer (sync version)
# ---------------------------------------------------------------------------
class HDF5Writer:
    """Manages per-episode HDF5 groups with buffered, chunked writes."""

    CHUNK_T = 16

    def __init__(self, path: str, action_dim: int, metadata: dict,
                 rdcc_nbytes: int = 64 * 1024 * 1024):
        self.path = path
        self.action_dim = action_dim
        self.metadata = metadata
        self.rdcc_nbytes = rdcc_nbytes
        self.h5file: Optional[h5py.File] = None
        self._episode_count = 0
        self._buffers: dict[int, dict[str, list]] = {}

    def __enter__(self):
        self.h5file = h5py.File(
            self.path, "w",
            rdcc_nbytes=self.rdcc_nbytes,
            rdcc_nslots=10007,
        )
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
        if env_id not in self._buffers:
            self._buffers[env_id] = {
                "latents": [], "actions": [], "rewards": [], "dones": [],
            }

    def append_frame(self, env_id: int, latent: np.ndarray,
                     action: np.ndarray, reward: float, done: bool):
        buf = self._buffers[env_id]
        buf["latents"].append(latent)
        buf["actions"].append(action)
        buf["rewards"].append(reward)
        buf["dones"].append(done)

    def append_batch(self, latents: np.ndarray, actions: np.ndarray,
                     rewards: np.ndarray, dones: np.ndarray, num_envs: int):
        for i in range(num_envs):
            self.ensure_episode(i)
            self.append_frame(i, latents[i], actions[i],
                              float(rewards[i]), bool(dones[i]))

    def finalize_episode(self, env_id: int, task: str, seed: int, completed: bool):
        buf = self._buffers.pop(env_id, None)
        if buf is None or len(buf["latents"]) == 0:
            return
        T = len(buf["latents"])
        grp = self.h5file.create_group(f"episode_{self._episode_count:04d}")
        lat_arr = np.stack(buf["latents"], axis=0).astype(np.float16)
        act_arr = np.stack(buf["actions"], axis=0).astype(np.float32)
        rew_arr = np.array(buf["rewards"], dtype=np.float32)
        done_arr = np.array(buf["dones"], dtype=bool)
        C = self.CHUNK_T
        grp.create_dataset("latents", data=lat_arr,
                           chunks=(min(C, T), *lat_arr.shape[1:]),
                           compression="lzf")
        grp.create_dataset("actions", data=act_arr,
                           chunks=(min(C, T), self.action_dim),
                           compression="lzf")
        grp.create_dataset("rewards", data=rew_arr,
                           chunks=(min(C, T),), compression="lzf")
        grp.create_dataset("dones", data=done_arr, chunks=(min(C, T),))
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
        for env_id in list(self._buffers.keys()):
            self.finalize_episode(env_id, task="unknown", seed=0, completed=False)
        self.flush()

    @property
    def episode_count(self) -> int:
        return self._episode_count


# ---------------------------------------------------------------------------
# 4b. Async HDF5 Writer (multiprocess queue)
# ---------------------------------------------------------------------------
_SENTINEL = None  # poison pill to stop writer process


def _hdf5_writer_process(path: str, action_dim: int, metadata: dict,
                         rdcc_nbytes: int, queue: mp.Queue):
    """Dedicated process that drains a queue and writes to HDF5."""
    writer = HDF5Writer(path, action_dim, metadata, rdcc_nbytes)
    writer.__enter__()
    try:
        while True:
            msg = queue.get()
            if msg is _SENTINEL:
                break
            cmd = msg[0]
            if cmd == "append":
                _, latents, actions, rewards, dones, num_envs = msg
                writer.append_batch(latents, actions, rewards, dones, num_envs)
            elif cmd == "finalize":
                _, env_id, task, seed, completed = msg
                writer.finalize_episode(env_id, task, seed, completed)
            elif cmd == "ensure":
                _, env_id = msg
                writer.ensure_episode(env_id)
            elif cmd == "flush":
                writer.flush()
    finally:
        writer.__exit__(None, None, None)


class AsyncHDF5Writer:
    """Wraps HDF5Writer in a separate process with a multiprocessing.Queue."""

    def __init__(self, path: str, action_dim: int, metadata: dict,
                 rdcc_nbytes: int = 64 * 1024 * 1024):
        self.queue: mp.Queue = mp.Queue(maxsize=256)
        self._proc: Optional[mp.Process] = None
        self._args = (path, action_dim, metadata, rdcc_nbytes, self.queue)

    def __enter__(self):
        self._proc = mp.Process(target=_hdf5_writer_process, args=self._args, daemon=True)
        self._proc.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._proc is not None:
            self.queue.put(_SENTINEL)
            self._proc.join(timeout=30)
            if self._proc.is_alive():
                self._proc.terminate()
            self._proc = None
        return False

    def ensure_episode(self, env_id: int):
        self.queue.put(("ensure", env_id))

    def append_batch(self, latents: np.ndarray, actions: np.ndarray,
                     rewards: np.ndarray, dones: np.ndarray, num_envs: int):
        self.queue.put(("append", latents, actions, rewards, dones, num_envs))

    def finalize_episode(self, env_id: int, task: str, seed: int, completed: bool):
        self.queue.put(("finalize", env_id, task, seed, completed))

    def flush(self):
        self.queue.put(("flush",))


# ---------------------------------------------------------------------------
# 5. LPS Benchmark
# ---------------------------------------------------------------------------
class LPSBenchmark:
    """Per-stage timing with CUDA events and NVTX ranges."""

    CUDA_STAGES = ("env_step", "preprocess", "vae_encode", "cpu_transfer")
    CPU_STAGES = ("hdf5_write",)
    ALL_STAGES = CUDA_STAGES + CPU_STAGES

    def __init__(self):
        self._start_events: dict[str, list] = {s: [] for s in self.CUDA_STAGES}
        self._end_events: dict[str, list] = {s: [] for s in self.CUDA_STAGES}
        self._cpu_times: dict[str, list[float]] = {s: [] for s in self.CPU_STAGES}
        self._total_start: float = 0.0
        self._total_end: float = 0.0

    def start_total(self):
        torch.cuda.synchronize()
        self._total_start = time.perf_counter()

    def stop_total(self):
        torch.cuda.synchronize()
        self._total_end = time.perf_counter()

    def cuda_start(self, stage: str):
        nvtx.range_push(stage)
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
        self._start_events[stage].append(ev)

    def cuda_stop(self, stage: str):
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

    def summarize(self, num_envs: int, num_steps: int) -> float:
        """Print LPS and per-stage breakdown. Returns LPS value."""
        torch.cuda.synchronize()
        wall = self._total_end - self._total_start
        lps = (num_envs * num_steps) / wall if wall > 0 else 0.0

        stage_ms: dict[str, float] = {}
        for stage in self.CUDA_STAGES:
            starts = self._start_events[stage]
            ends = self._end_events[stage]
            stage_ms[stage] = sum(s.elapsed_time(e) for s, e in zip(starts, ends))
        for stage in self.CPU_STAGES:
            stage_ms[stage] = sum(self._cpu_times[stage]) * 1000.0

        total_staged = sum(stage_ms.values())

        print("\n" + "=" * 60)
        print(f"LPS Benchmark  --  {lps:,.0f} latents/sec  ({wall:.2f}s wall)")
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
        return lps


# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------
_shutdown_flag = False


def _signal_handler(signum, frame):
    global _shutdown_flag
    _shutdown_flag = True


# ---------------------------------------------------------------------------
# Main ingestion loop (v2 — pipelined)
# ---------------------------------------------------------------------------
def run(cfg: IngestConfig):
    """Pipelined ingestion: env.step on default stream, VAE on encode stream.

    Double-buffer: while env steps on frame_buf[0], VAE encodes frame_buf[1]
    from the previous step, and vice versa.
    """
    global _shutdown_flag
    _shutdown_flag = False

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    torch.manual_seed(cfg.seed)

    # Double-buffered frame buffers for pipeline overlap
    frame_bufs = [
        torch.empty(cfg.num_envs, 3, OBS_H, OBS_W, dtype=torch.float16, device="cuda"),
        torch.empty(cfg.num_envs, 3, OBS_H, OBS_W, dtype=torch.float16, device="cuda"),
    ]

    encoder = CosmosLatentEncoder(cfg.cosmos_ckpt)
    bench = LPSBenchmark()

    # Choose sync or async writer
    WriterCls = AsyncHDF5Writer if cfg.async_writer else HDF5Writer
    writer_kwargs = dict(
        path=cfg.hdf5_path,
        action_dim=0,  # placeholder, set after env creation
        metadata={
            "cosmos_model": "nvidia/Cosmos-0.1-Tokenizer-CI16x16",
            "compression_ratio": 256,
            "obs_resolution": OBS_H,
        },
        rdcc_nbytes=cfg.rdcc_nbytes,
    )

    with ManiSkillCollector(cfg) as collector:
        writer_kwargs["action_dim"] = collector.action_dim
        with WriterCls(**writer_kwargs) as writer:
            obs, _ = collector.reset()
            for i in range(cfg.num_envs):
                writer.ensure_episode(i)

            completed_episodes = 0
            step_count = 0
            buf_idx = 0  # current buffer index
            pending_latents = None  # latents from previous step (still encoding)

            bench.start_total()

            for step in range(cfg.max_total_steps):
                if _shutdown_flag or completed_episodes >= cfg.max_episodes:
                    break

                cur_buf = frame_bufs[buf_idx]
                prev_buf_idx = 1 - buf_idx

                # --- 1. Step envs on default stream ---
                bench.cuda_start("env_step")
                actions = collector.sample_random_actions()
                obs, rewards, terms, truncs, infos = collector.step(actions)
                bench.cuda_stop("env_step")

                # --- 2. Preprocess into current buffer ---
                bench.cuda_start("preprocess")
                collector.extract_rgb(obs, cur_buf)
                bench.cuda_stop("preprocess")

                # --- 3. If there's a pending encode from prev step, sync and write ---
                if pending_latents is not None:
                    encoder.sync()  # wait for previous encode to finish

                    bench.cuda_start("cpu_transfer")
                    latents_cpu = pending_latents.cpu().numpy()
                    actions_cpu = prev_actions.cpu().numpy()
                    rewards_cpu = prev_rewards.cpu().numpy()
                    dones_cpu = prev_dones.cpu().numpy()
                    bench.cuda_stop("cpu_transfer")

                    bench.cpu_start("hdf5_write")
                    writer.append_batch(latents_cpu, actions_cpu, rewards_cpu,
                                        dones_cpu, num_envs=cfg.num_envs)
                    bench.cpu_stop("hdf5_write")

                    step_count += 1

                    # Handle episode resets from previous step
                    done_indices = prev_dones.nonzero(as_tuple=False).squeeze(-1)
                    if done_indices.numel() > 0:
                        done_list = done_indices.cpu().tolist()
                        for env_id in done_list:
                            writer.finalize_episode(
                                env_id=env_id, task=cfg.task,
                                seed=cfg.seed,
                                completed=bool(prev_terms[env_id].item()),
                            )
                            completed_episodes += 1
                            writer.ensure_episode(env_id)
                            if completed_episodes % cfg.flush_every == 0:
                                writer.flush()
                            if completed_episodes >= cfg.max_episodes:
                                break

                # --- 4. Launch VAE encode on encode stream (async) ---
                bench.cuda_start("vae_encode")
                pending_latents = encoder.encode(cur_buf)
                bench.cuda_stop("vae_encode")

                # Save current step's metadata for next iteration's write
                prev_actions = actions
                prev_rewards = rewards
                prev_terms = terms
                prev_dones = terms | truncs
                buf_idx = 1 - buf_idx  # swap buffers

            # --- Drain final pending encode ---
            if pending_latents is not None:
                encoder.sync()
                latents_cpu = pending_latents.cpu().numpy()
                actions_cpu = prev_actions.cpu().numpy()
                rewards_cpu = prev_rewards.cpu().numpy()
                dones_cpu = prev_dones.cpu().numpy()
                writer.append_batch(latents_cpu, actions_cpu, rewards_cpu,
                                    dones_cpu, num_envs=cfg.num_envs)
                step_count += 1
                done_indices = prev_dones.nonzero(as_tuple=False).squeeze(-1)
                if done_indices.numel() > 0:
                    for env_id in done_indices.cpu().tolist():
                        writer.finalize_episode(
                            env_id=env_id, task=cfg.task,
                            seed=cfg.seed,
                            completed=bool(prev_terms[env_id].item()),
                        )
                        completed_episodes += 1
                        writer.ensure_episode(env_id)

            bench.stop_total()
            lps = bench.summarize(num_envs=cfg.num_envs, num_steps=step_count)

            ep_count = writer.episode_count if hasattr(writer, 'episode_count') else "N/A"
            print(f"\nCompleted {completed_episodes} episodes, "
                  f"{ep_count} written to {cfg.hdf5_path}")

    return lps


# ---------------------------------------------------------------------------
# Latent distribution verification
# ---------------------------------------------------------------------------
def verify_latents(ckpt_dir: str, num_samples: int = 8):
    encoder = CosmosLatentEncoder(ckpt_dir)
    fake_rgb = torch.rand(num_samples, 3, OBS_H, OBS_W,
                          dtype=torch.float16, device="cuda")
    latents = encoder.encode(fake_rgb)
    encoder.sync()

    print(f"\n{'='*50}")
    print("Cosmos-Tokenizer-CI16x16 Latent Verification")
    print(f"{'='*50}")
    print(f"Input shape:  {list(fake_rgb.shape)}  dtype={fake_rgb.dtype}")
    print(f"Output shape: {list(latents.shape)}  dtype={latents.dtype}")
    print(f"Expected:     [{num_samples}, {LATENT_C}, {LATENT_H}, {LATENT_W}]")
    assert latents.shape == (num_samples, LATENT_C, LATENT_H, LATENT_W), \
        f"Shape mismatch: {latents.shape}"
    print(f"  mean:  {latents.float().mean().item():.4f}")
    print(f"  std:   {latents.float().std().item():.4f}")
    print(f"  min:   {latents.float().min().item():.4f}")
    print(f"  max:   {latents.float().max().item():.4f}")
    print("Shape check PASSED")
    print(f"{'='*50}\n")
    return latents


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="ManiSkill -> Cosmos-VAE -> HDF5 pipeline")
    p.add_argument("--task", type=str, default="PickCube-v1")
    p.add_argument("--num_envs", type=int, default=128)
    p.add_argument("--max_episodes", type=int, default=100)
    p.add_argument("--max_steps", type=int, default=200)
    p.add_argument("--hdf5_path", type=str, default="trajectories.h5")
    p.add_argument("--cosmos_ckpt", type=str,
                    default="pretrained_ckpts/Cosmos-Tokenizer-CI16x16")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--flush_every", type=int, default=50)
    p.add_argument("--rdcc_nbytes", type=int, default=64 * 1024 * 1024)
    p.add_argument("--rgb_obs", action="store_true")
    p.add_argument("--no_async_writer", action="store_true",
                    help="Disable async HDF5 writer (use sync instead)")
    p.add_argument("--verify_only", action="store_true")
    args = p.parse_args()
    cfg = IngestConfig(**{k: v for k, v in vars(args).items()
                          if k not in ("verify_only", "no_async_writer")})
    cfg.async_writer = not args.no_async_writer
    return cfg, args.verify_only


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    cfg, verify_only = parse_args()
    if verify_only:
        verify_latents(cfg.cosmos_ckpt)
    else:
        verify_latents(cfg.cosmos_ckpt)
        run(cfg)
