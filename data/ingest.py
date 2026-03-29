"""ManiSkill 3.0 Trajectory Ingestion Pipeline.

Collects robotics trajectories from ManiSkill GPU-parallelized environments,
compresses observations into latents via Cosmos-VAE, and stores them in HDF5
for efficient sliding-window training retrieval.

Usage:
    python data/ingest.py --task PickCube-v1 --num_envs 128 --max_episodes 100000

Optimizations (v2):
  - Double-buffered VAE on a separate CUDA stream (overlaps encode with env.step)
  - Async HDF5 writer on a dedicated process via multiprocessing.Queue
  - rdcc_nbytes=64MB chunk cache for sequential write performance
  - Increased default num_envs=64 to better amortize physics step cost

Rendering: obs_mode="state" + render_mode="rgb_array" + render_backend="cpu".
RGB frames come from env.render() after each step, not the obs dict. This avoids
the Vulkan-CUDA interop crash on RTX 50-series Windows (same approach as
verify_env.py which is confirmed working).
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
    max_steps: int = 400
    hdf5_path: str = "trajectories.h5"
    cosmos_ckpt: str = "pretrained_ckpts/Cosmos-Tokenizer-CI16x16"
    seed: int = 42
    flush_every: int = 50
    rdcc_nbytes: int = 64 * 1024 * 1024
    async_writer: bool = True  # use multiprocess HDF5 writer
    noise_scale: float = 0.13
    control_mode: str = "pd_ee_delta_pos"
    sim_backend: str = "gpu"  # "gpu" or "cpu"; cpu requires num_envs=1 per process
    robot_init_qpos_noise: float = 0.10   # std (rad) of shoulder/elbow noise (wrist is fixed)
    cube_spawn_half_size: float = 0.15    # half-side (m) of cube/goal XY spawn region

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
    """Wraps ManiSkill env with CPU renderer for real RGB obs.

    Uses obs_mode="state" + render_mode="rgb_array" + render_backend="cpu" to
    avoid the Vulkan-CUDA interop crash on RTX 50-series Windows. RGB frames are
    obtained via env.render() after each step rather than from the obs dict.
    """

    def __init__(self, cfg: IngestConfig):
        self.cfg = cfg
        self.env: Optional[gym.Env] = None

    def __enter__(self):
        kwargs = dict(
            num_envs=self.cfg.num_envs,
            obs_mode="state",
            render_mode="rgb_array",
            render_backend="cpu",
            control_mode=self.cfg.control_mode,
            max_episode_steps=self.cfg.max_steps,
            robot_init_qpos_noise=self.cfg.robot_init_qpos_noise,
        )
        if self.cfg.sim_backend == "cpu":
            kwargs["sim_backend"] = "cpu"
        self.env = gym.make(self.cfg.task, **kwargs)
        self.env.unwrapped.cube_spawn_half_size = self.cfg.cube_spawn_half_size
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.env is not None:
            self.env.close()
            self.env = None
        return False

    def reset(self):
        result = self.env.reset(seed=self.cfg.seed)
        self._fix_wrist_orientation()
        return result

    def _fix_wrist_orientation(self):
        """Reset Panda wrist joints (4-6) to canonical values after qpos noise.
        Keeps shoulder/elbow noise (arm position diversity) but ensures the
        gripper always points straight down for reliable grasping."""
        uwenv = self.env.unwrapped
        if getattr(uwenv, "robot_uids", None) not in ("panda", "panda_wristcam", None):
            return  # only needed for Panda
        # Canonical wrist qpos from TableSceneBuilder
        _WRIST = torch.tensor([0.0, np.pi * 3 / 4, np.pi / 4],
                               dtype=torch.float32, device=uwenv.device)
        qpos = uwenv.agent.robot.get_qpos().clone()  # [N, 9]
        qpos[:, 4:7] = _WRIST
        uwenv.agent.robot.set_qpos(qpos)

    def step(self, actions: torch.Tensor):
        return self.env.step(actions)

    def extract_rgb(self, obs, frame_buf: torch.Tensor):
        """Render RGB via env.render() and fill frame_buf [N,3,H,W] float16.

        obs is unused — kept for API compatibility with the main loop.
        env.render() returns [N,H,W,3] uint8 via the CPU renderer.
        Resizes to (OBS_H, OBS_W) if the render resolution differs.
        """
        frame = self.env.render()  # [N, H, W, 3] uint8
        if not isinstance(frame, torch.Tensor):
            frame = torch.from_numpy(np.asarray(frame))
        # [N,H,W,3] → [N,3,H,W] float32 in [0,1]
        frame = frame[..., :3].permute(0, 3, 1, 2).float().div_(255.0)
        if frame.shape[-2:] != (OBS_H, OBS_W):
            frame = torch.nn.functional.interpolate(
                frame, size=(OBS_H, OBS_W), mode="bilinear", align_corners=False,
            )
        frame_buf.copy_(frame.to(torch.float16))

    @property
    def action_dim(self) -> int:
        return int(np.prod(self.env.unwrapped.single_action_space.shape))


# ---------------------------------------------------------------------------
# 3. Guided Policy (distance heuristic + 4-phase state machine)
# ---------------------------------------------------------------------------
class GuidedPolicy:
    """GPU-batched guided trajectory policy for PickCube-v1.

    Requires pd_ee_delta_pos control mode (pinocchio handles IK internally).
    Action space: [dx, dy, dz, gripper] — 4D world-frame EE deltas.

    Phases (int32 per env on CUDA):
        0 = APPROACH  — move EE to (cube_xy, cube_z + HOVER_HEIGHT)
        1 = DESCEND   — move EE to (cube_xy, cube_z + NEAR_HEIGHT)
        2 = GRASP     — hold position, close gripper for GRASP_STEPS steps
        3 = LIFT      — move EE toward goal_pos (actual task goal marker)
        4 = HOLD      — hold position at goal, gripper closed, until static → success
    """

    PHASE_APPROACH = 0
    PHASE_DESCEND  = 1
    PHASE_GRASP    = 2
    PHASE_LIFT     = 3
    PHASE_HOLD     = 4

    HOVER_HEIGHT = 0.10    # metres above cube for APPROACH target
    NEAR_HEIGHT  = 0.00    # metres above cube for DESCEND target (TCP at cube centre)

    GRASP_STEPS = 30
    HOLD_STEPS  = 60

    APPROACH_THRESH = 0.04   # metres — triggers APPROACH→DESCEND early
    DESCEND_THRESH  = 0.008  # metres — triggers DESCEND→GRASP early
    LIFT_THRESH     = 0.05   # metres z-distance — triggers LIFT→HOLD at goal height

    EE_GAIN    = 0.08   # max EE displacement per step (clamps delta before sending)
    GRIPPER_DIM = 3     # index of gripper action in 4D action vector

    def __init__(
        self,
        env,
        num_envs: int,
        noise_scale: float = 0.05,
        gripper_noise_scale: float = 0.02,
        device: str = "cuda",
    ):
        self.env = env
        self.num_envs = num_envs
        self.noise_scale = noise_scale
        self.gripper_noise_scale = gripper_noise_scale
        self.device = torch.device(device)

        self.phases      = torch.zeros(num_envs, dtype=torch.int32, device=self.device)
        self.phase_steps = torch.zeros(num_envs, dtype=torch.int32, device=self.device)

        action_space = env.unwrapped.single_action_space
        action_dim   = int(np.prod(action_space.shape))
        if action_dim != 4:
            raise ValueError(
                f"GuidedPolicy requires pd_ee_delta_pos (action_dim=4), "
                f"got action_dim={action_dim}. Pass control_mode='pd_ee_delta_pos' to gym.make."
            )
        self._lo = torch.as_tensor(action_space.low,  dtype=torch.float32, device=self.device)
        self._hi = torch.as_tensor(action_space.high, dtype=torch.float32, device=self.device)

        # Cache gripper bounds as Python floats (only .item() calls in this class)
        self._gripper_open  = float(self._hi[self.GRIPPER_DIM].item())
        self._gripper_close = float(self._lo[self.GRIPPER_DIM].item())

        # Only GRASP and HOLD have real budgets; APPROACH/DESCEND/LIFT use proximity triggers
        _INF = 10_000
        self._phase_budgets = torch.tensor(
            [_INF, _INF, self.GRASP_STEPS, _INF, self.HOLD_STEPS],
            dtype=torch.int32, device=self.device,
        )

        # Probe goal position attribute (PickCube-v1 exposes goal_site)
        self._goal_attr = "goal_site" if hasattr(env.unwrapped, "goal_site") else None

        # Pre-allocated buffers — zero allocation in __call__
        self._action_buf        = torch.zeros(num_envs, action_dim, dtype=torch.float32, device=self.device)
        self._target_buf        = torch.zeros(num_envs, 3, dtype=torch.float32, device=self.device)
        self._noise_buf_ee      = torch.zeros(num_envs, 3, dtype=torch.float32, device=self.device)
        self._noise_buf_gripper = torch.zeros(num_envs, dtype=torch.float32, device=self.device)

        # Probe attribute names once (CPU-side, before hot path)
        self._cube_attr = self._probe_cube_attribute()

        # Verify EE path exists
        uwenv = env.unwrapped
        if not hasattr(uwenv, "agent") or not hasattr(uwenv.agent, "tcp"):
            raise AttributeError(
                "env.unwrapped.agent.tcp not found — check ManiSkill version / task"
            )

        print(f"[GuidedPolicy] pd_ee_delta_pos  action_dim={action_dim}")

    def _probe_cube_attribute(self) -> str:
        uwenv = self.env.unwrapped
        candidates = ["cube", "obj", "object", "target_object", "box"]
        for name in candidates:
            if hasattr(uwenv, name):
                return name
        raise AttributeError(
            f"Cannot find cube attribute on env.unwrapped. Tried: {candidates}. "
            f"Add the correct name to GuidedPolicy._probe_cube_attribute."
        )

    def _get_ee_and_cube_pos(self):
        """Returns (ee_pos, cube_pos, goal_pos) each [N, 3] float32 on self.device."""
        uwenv    = self.env.unwrapped
        ee_pos   = uwenv.agent.tcp.pose.p.to(self.device)
        cube_pos = getattr(uwenv, self._cube_attr).pose.p.to(self.device)
        if self._goal_attr is not None:
            goal_pos = uwenv.goal_site.pose.p.to(self.device)
        else:
            # Fallback: target 0.2m above cube spawn if no goal marker
            goal_pos = cube_pos.clone()
            goal_pos[:, 2] = goal_pos[:, 2] + 0.20
        return ee_pos, cube_pos, goal_pos

    def _advance_phases(self, ee_pos: torch.Tensor, cube_pos: torch.Tensor,
                        goal_pos: torch.Tensor):
        """Update self.phases and self.phase_steps inplace using only torch.where."""
        hover_target = cube_pos.clone()
        hover_target[:, 2] = cube_pos[:, 2] + self.HOVER_HEIGHT
        near_target  = cube_pos.clone()
        near_target[:, 2]  = cube_pos[:, 2] + self.NEAR_HEIGHT

        phase_is_0 = (self.phases == self.PHASE_APPROACH)
        active_target = torch.where(phase_is_0.unsqueeze(1), hover_target, near_target)
        dist = torch.norm(ee_pos - active_target, dim=-1)  # [N]

        budget = self._phase_budgets[self.phases]
        budget_exceeded = self.phase_steps >= budget

        thresh = torch.where(
            phase_is_0,
            torch.full_like(dist, self.APPROACH_THRESH),
            torch.full_like(dist, self.DESCEND_THRESH),
        )
        in_proximity_phase = (self.phases == self.PHASE_APPROACH) | \
                             (self.phases == self.PHASE_DESCEND)
        proximity_trigger  = in_proximity_phase & (dist < thresh)

        # LIFT → HOLD when EE z reaches goal z (vertical-only lift, safe for grip)
        z_dist_to_goal = torch.abs(ee_pos[:, 2] - goal_pos[:, 2])
        lift_trigger   = (self.phases == self.PHASE_LIFT) & (z_dist_to_goal < self.LIFT_THRESH)

        should_advance = budget_exceeded | proximity_trigger | lift_trigger

        next_phase = (self.phases + 1).clamp_(max=self.PHASE_HOLD)
        self.phases      = torch.where(should_advance, next_phase, self.phases)
        self.phase_steps.mul_((~should_advance).int()).add_(1)

    def _compute_actions(self, ee_pos: torch.Tensor, cube_pos: torch.Tensor,
                         goal_pos: torch.Tensor):
        """Fill self._action_buf inplace with joint delta actions."""
        # --- Target selection ---
        self._target_buf.copy_(cube_pos)
        approach_z = cube_pos[:, 2] + self.HOVER_HEIGHT
        descend_z  = cube_pos[:, 2] + self.NEAR_HEIGHT

        phase_is_0 = (self.phases == self.PHASE_APPROACH).unsqueeze(1)
        phase_is_3 = (self.phases == self.PHASE_LIFT).unsqueeze(1)
        phase_is_4 = (self.phases == self.PHASE_HOLD).unsqueeze(1)

        target_z_012 = torch.where(phase_is_0.squeeze(1), approach_z, descend_z)
        self._target_buf[:, 2] = target_z_012

        # LIFT: straight up to goal z, keep EE xy fixed — safe for grip stability
        lift_target = ee_pos.clone()
        lift_target[:, 2] = goal_pos[:, 2]
        # HOLD: target full goal_pos (horizontal slide into place), no noise → goes static
        target = torch.where(phase_is_3, lift_target, self._target_buf)
        target = torch.where(phase_is_4, goal_pos,    target)

        # --- World-frame EE delta, clamped to EE_GAIN ---
        delta = (target - ee_pos).clamp_(-self.EE_GAIN, self.EE_GAIN)  # [N, 3]

        # --- pd_ee_delta_pos: dims 0-2 are directly [dx, dy, dz] in world frame ---
        self._action_buf.zero_()
        self._action_buf[:, :3].copy_(delta)

        # --- Gripper: open for phases 0,1; close for phases 2,3,4 ---
        is_closed   = ((self.phases == self.PHASE_GRASP) |
                       (self.phases == self.PHASE_LIFT)  |
                       (self.phases == self.PHASE_HOLD)).float()
        gripper_cmd = self._gripper_open + \
                      (self._gripper_close - self._gripper_open) * is_closed
        self._action_buf[:, self.GRIPPER_DIM].copy_(gripper_cmd)

        return self._action_buf

    def _inject_noise(self):
        """Add Gaussian noise inplace, then clamp to action bounds."""
        self._noise_buf_ee.normal_(0.0, self.noise_scale)
        self._action_buf[:, :3].add_(self._noise_buf_ee)
        self._noise_buf_gripper.normal_(0.0, self.gripper_noise_scale)
        self._action_buf[:, self.GRIPPER_DIM].add_(self._noise_buf_gripper)
        self._action_buf.clamp_(self._lo, self._hi)

    def reset_done_envs(self, dones: torch.Tensor):
        """Reset phase to APPROACH for envs where dones=True. GPU-safe, no .item()."""
        not_done = (~dones.to(self.device)).int()
        self.phases.mul_(not_done)
        self.phase_steps.mul_(not_done)

    def __call__(self) -> torch.Tensor:
        with nvtx.range("GuidedPolicy"):
            ee_pos, cube_pos, goal_pos = self._get_ee_and_cube_pos()
            self._advance_phases(ee_pos, cube_pos, goal_pos)
            self._compute_actions(ee_pos, cube_pos, goal_pos)
            self._inject_noise()
            # Expose state for the data-collection loop to store per step
            self.last_ee_pos   = ee_pos    # [N, 3] absolute EE position
            self.last_cube_pos = cube_pos  # [N, 3] cube position
            return self._action_buf


# ---------------------------------------------------------------------------
# 4. Cosmos Latent Encoder (with separate CUDA stream)
# ---------------------------------------------------------------------------
class CosmosLatentEncoder:
    """Cosmos-Tokenizer-CI16x16 encoder with optional CUDA stream pipelining.

    Input:  [B, 3, 128, 128] float16 in [0,1]
    Output: [B, 16, 8, 8]    float32
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
        return latents.to(torch.float32)

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
                "latents": [], "actions": [], "rewards": [],
                "terminated": [], "truncated": [], "success": [],
                "ee_pos": [], "cube_pos": [], "phase": [],
            }

    def append_frame(self, env_id: int, latent: np.ndarray,
                     action: np.ndarray, reward: float,
                     terminated: bool, truncated: bool, success: bool,
                     ee_pos: np.ndarray, cube_pos: np.ndarray, phase: int):
        buf = self._buffers[env_id]
        buf["latents"].append(latent)
        buf["actions"].append(action)
        buf["rewards"].append(reward)
        buf["terminated"].append(terminated)
        buf["truncated"].append(truncated)
        buf["success"].append(success)
        buf["ee_pos"].append(ee_pos)
        buf["cube_pos"].append(cube_pos)
        buf["phase"].append(phase)

    def append_batch(self, latents: np.ndarray, actions: np.ndarray,
                     rewards: np.ndarray, terminated: np.ndarray,
                     truncated: np.ndarray, success: np.ndarray,
                     ee_pos: np.ndarray, cube_pos: np.ndarray,
                     phases: np.ndarray, num_envs: int):
        for i in range(num_envs):
            self.ensure_episode(i)
            self.append_frame(i, latents[i], actions[i],
                              float(rewards[i]),
                              bool(terminated[i]), bool(truncated[i]), bool(success[i]),
                              ee_pos[i], cube_pos[i], int(phases[i]))

    def finalize_episode(self, env_id: int, task: str, seed: int, completed: bool):
        buf = self._buffers.pop(env_id, None)
        if buf is None or len(buf["latents"]) == 0:
            return
        T = len(buf["latents"])
        grp = self.h5file.create_group(f"episode_{self._episode_count:04d}")
        lat_arr      = np.stack(buf["latents"],    axis=0).astype(np.float32)
        act_arr      = np.stack(buf["actions"],    axis=0).astype(np.float32)
        rew_arr      = np.array(buf["rewards"],    dtype=np.float32)
        term_arr     = np.array(buf["terminated"], dtype=bool)
        trunc_arr    = np.array(buf["truncated"],  dtype=bool)
        succ_arr     = np.array(buf["success"],    dtype=bool)
        ee_pos_arr   = np.stack(buf["ee_pos"],     axis=0).astype(np.float32)
        cube_pos_arr = np.stack(buf["cube_pos"],   axis=0).astype(np.float32)
        phase_arr    = np.array(buf["phase"],      dtype=np.int8)
        C = self.CHUNK_T
        grp.create_dataset("latents", data=lat_arr,
                           chunks=(min(C, T), *lat_arr.shape[1:]),
                           compression="lzf")
        grp.create_dataset("actions", data=act_arr,
                           chunks=(min(C, T), self.action_dim),
                           compression="lzf")
        grp.create_dataset("rewards",    data=rew_arr,   chunks=(min(C, T),), compression="lzf")
        grp.create_dataset("terminated", data=term_arr,  chunks=(min(C, T),))
        grp.create_dataset("truncated",  data=trunc_arr, chunks=(min(C, T),))
        grp.create_dataset("success",    data=succ_arr,  chunks=(min(C, T),))
        grp.create_dataset("ee_pos",   data=ee_pos_arr,   chunks=(min(C, T), 3),
                           compression="lzf")
        grp.create_dataset("cube_pos", data=cube_pos_arr, chunks=(min(C, T), 3),
                           compression="lzf")
        grp.create_dataset("phase",    data=phase_arr,    chunks=(min(C, T),),
                           compression="lzf")
        grp.attrs["episode_id"] = self._episode_count
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
                _, latents, actions, rewards, terminated, truncated, success, ee_pos, cube_pos, phases, num_envs = msg
                writer.append_batch(latents, actions, rewards, terminated, truncated, success,
                                    ee_pos, cube_pos, phases, num_envs)
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
                     rewards: np.ndarray, terminated: np.ndarray,
                     truncated: np.ndarray, success: np.ndarray,
                     ee_pos: np.ndarray, cube_pos: np.ndarray,
                     phases: np.ndarray, num_envs: int):
        self.queue.put(("append", latents, actions, rewards, terminated, truncated, success,
                        ee_pos, cube_pos, phases, num_envs))

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

            policy = GuidedPolicy(
                env=collector.env,
                num_envs=cfg.num_envs,
                noise_scale=cfg.noise_scale,
                device="cpu" if cfg.sim_backend == "cpu" else "cuda",
            )
            obs, _ = collector.reset()

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
                actions = policy()
                # Capture state immediately after policy() before any phase reset
                cur_ee_pos   = policy.last_ee_pos.cpu()    # [N, 3]
                cur_cube_pos = policy.last_cube_pos.cpu()  # [N, 3]
                cur_phases   = policy.phases.cpu()         # [N]
                obs, rewards, terms, truncs, infos = collector.step(actions)
                # Extract per-env success flag from info dict
                raw_success = infos.get("success", False)
                if isinstance(raw_success, torch.Tensor):
                    cur_success = raw_success.to(torch.bool).cpu()
                else:
                    cur_success = torch.zeros(cfg.num_envs, dtype=torch.bool)
                bench.cuda_stop("env_step")

                # --- 2. Preprocess into current buffer ---
                bench.cuda_start("preprocess")
                collector.extract_rgb(obs, cur_buf)
                bench.cuda_stop("preprocess")

                # --- 3. If there's a pending encode from prev step, sync and write ---
                if pending_latents is not None:
                    encoder.sync()  # wait for previous encode to finish

                    bench.cuda_start("cpu_transfer")
                    latents_cpu  = pending_latents.cpu().numpy()
                    actions_cpu  = prev_actions.cpu().numpy()
                    rewards_cpu  = prev_rewards.cpu().numpy()
                    terms_cpu    = prev_terms.cpu().numpy()
                    truncs_cpu   = prev_truncs.cpu().numpy()
                    success_cpu  = prev_success.numpy()
                    ee_pos_cpu   = prev_ee_pos.numpy()
                    cube_pos_cpu = prev_cube_pos.numpy()
                    phases_cpu   = prev_phases.numpy()
                    bench.cuda_stop("cpu_transfer")

                    bench.cpu_start("hdf5_write")
                    writer.append_batch(latents_cpu, actions_cpu, rewards_cpu,
                                        terms_cpu, truncs_cpu, success_cpu,
                                        ee_pos_cpu, cube_pos_cpu,
                                        phases_cpu, num_envs=cfg.num_envs)
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
                prev_actions   = actions
                prev_rewards   = rewards
                prev_terms     = terms
                prev_truncs    = truncs
                prev_success   = cur_success
                prev_dones     = terms | truncs
                prev_ee_pos    = cur_ee_pos
                prev_cube_pos  = cur_cube_pos
                prev_phases    = cur_phases
                policy.reset_done_envs(prev_dones.bool())
                buf_idx = 1 - buf_idx  # swap buffers

            # --- Drain final pending encode ---
            if pending_latents is not None:
                encoder.sync()
                latents_cpu  = pending_latents.cpu().numpy()
                actions_cpu  = prev_actions.cpu().numpy()
                rewards_cpu  = prev_rewards.cpu().numpy()
                terms_cpu    = prev_terms.cpu().numpy()
                truncs_cpu   = prev_truncs.cpu().numpy()
                success_cpu  = prev_success.numpy()
                ee_pos_cpu   = prev_ee_pos.numpy()
                cube_pos_cpu = prev_cube_pos.numpy()
                phases_cpu   = prev_phases.numpy()
                writer.append_batch(latents_cpu, actions_cpu, rewards_cpu,
                                    terms_cpu, truncs_cpu, success_cpu,
                                    ee_pos_cpu, cube_pos_cpu,
                                    phases_cpu, num_envs=cfg.num_envs)
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

            print(f"\nCompleted {completed_episodes} episodes written to {cfg.hdf5_path}")

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
    p.add_argument("--no_async_writer", action="store_true",
                    help="Disable async HDF5 writer (use sync instead)")
    p.add_argument("--noise_scale", type=float, default=0.05,
                    help="Gaussian noise std on EE action dims")
    p.add_argument("--control_mode", type=str, default="pd_ee_delta_pos",
                    choices=["pd_ee_delta_pos"],
                    help="Robot controller: pd_ee_delta_pos (pinocchio required)")
    p.add_argument("--sim_backend", type=str, default="gpu", choices=["gpu", "cpu"],
                    help="Physics backend. Use 'cpu' if cuda.dll unavailable (num_envs=1 only).")
    p.add_argument("--robot_init_qpos_noise", type=float, default=0.02,
                    help="Std (rad) of joint-angle noise applied to robot at episode start.")
    p.add_argument("--cube_spawn_half_size", type=float, default=0.10,
                    help="Half-side (m) of the XY region where cube and goal are spawned.")
    p.add_argument("--verify_only", action="store_true")
    args = p.parse_args()
    cfg = IngestConfig(**{k: v for k, v in vars(args).items()
                          if k not in ("verify_only", "no_async_writer", "guided")})
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
