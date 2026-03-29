"""Parallel trajectory collection with a single shared Cosmos encoder.

Architecture
------------
  EncoderServer  (1 process, GPU): owns the Cosmos model, encodes frames in
                 batches as they arrive from workers.
  Worker         (N processes, CPU): runs ManiSkill CPU-sim + GuidedPolicy,
                 sends RGB frames to the encoder server, waits for latents,
                 writes latents+actions+rewards to an HDF5 shard.

GPU memory: only 1 Cosmos model instance regardless of worker count.

Usage
-----
    python scripts/collect_parallel.py
    python scripts/collect_parallel.py --num_workers 16 --episodes_per_worker 1250

After collection, merge shards:
    python scripts/merge_shards.py --shard_dir trajectories_shards --output trajectories_20k.h5
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import torch

# ── import shared classes from data/ingest.py ─────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.ingest import (
    CosmosLatentEncoder,
    GuidedPolicy,
    HDF5Writer,
    IngestConfig,
    ManiSkillCollector,
    OBS_H,
    OBS_W,
    ensure_cosmos_weights,
)

_SENTINEL = None   # poison pill: worker sends this when it is done


# ─────────────────────────────────────────────────────────────────────────────
# Encoder server
# ─────────────────────────────────────────────────────────────────────────────
def encoder_server(
    cosmos_ckpt: str,
    frame_queue: mp.Queue,
    result_queues: list[mp.Queue],
    num_workers: int,
    batch_timeout: float = 0.02,
) -> None:
    """GPU process: drain frame_queue, encode in batches, fan out latents."""
    encoder = CosmosLatentEncoder(cosmos_ckpt)
    print("[EncoderServer] ready", flush=True)

    done_workers = 0
    while done_workers < num_workers:
        # --- collect a batch ------------------------------------------------
        batch: list[tuple[int, np.ndarray]] = []

        # blocking get for the first item
        try:
            item = frame_queue.get(timeout=5.0)
        except Exception:
            continue

        if item is _SENTINEL:
            done_workers += 1
            continue
        batch.append(item)

        # non-blocking drain up to num_workers frames
        deadline = time.perf_counter() + batch_timeout
        while time.perf_counter() < deadline:
            try:
                item = frame_queue.get_nowait()
                if item is _SENTINEL:
                    done_workers += 1
                else:
                    batch.append(item)
            except Exception:
                break

        if not batch:
            continue

        # --- encode ---------------------------------------------------------
        worker_ids = [b[0] for b in batch]
        frames_np  = np.stack([b[1] for b in batch], axis=0)   # [B,3,H,W] f16
        frames_t   = torch.from_numpy(frames_np).cuda()
        latents_t  = encoder.encode(frames_t)
        encoder.sync()
        latents_np = latents_t.cpu().numpy()                    # [B,16,8,8] f32

        # --- fan out --------------------------------------------------------
        for i, wid in enumerate(worker_ids):
            result_queues[wid].put(latents_np[i])

    print("[EncoderServer] all workers done, exiting", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Worker
# ─────────────────────────────────────────────────────────────────────────────
def worker(
    worker_id: int,
    cfg: IngestConfig,
    frame_queue: mp.Queue,
    result_queue: mp.Queue,
) -> None:
    """CPU-sim process: physics → RGB → encoder → HDF5."""
    print(f"[Worker {worker_id:02d}] started  seed={cfg.seed}", flush=True)

    # pre-allocate frame buffer; CPU sim workers don't use CUDA for physics
    fb_device = "cpu" if cfg.sim_backend == "cpu" else "cuda"
    frame_buf = torch.empty(1, 3, OBS_H, OBS_W, dtype=torch.float16, device=fb_device)

    from datetime import datetime, timezone
    metadata = {
        "cosmos_model": "nvidia/Cosmos-0.1-Tokenizer-CI16x16",
        "compression_ratio": 256,
        "obs_resolution": OBS_H,
        "worker_id": worker_id,
    }

    completed = 0

    with ManiSkillCollector(cfg) as collector:
        with HDF5Writer(cfg.hdf5_path, collector.action_dim, metadata) as writer:
            policy = GuidedPolicy(
                env=collector.env,
                num_envs=1,
                noise_scale=cfg.noise_scale,
                device="cpu" if cfg.sim_backend == "cpu" else "cuda",
            )
            obs, _ = collector.reset()
            writer.ensure_episode(0)

            for _step in range(cfg.max_total_steps):
                if completed >= cfg.max_episodes:
                    break

                # 1. step
                actions                            = policy()
                # Capture proprioceptive state right after policy() call
                ee_pos_np   = policy.last_ee_pos[0].cpu().numpy()    # [3]
                cube_pos_np = policy.last_cube_pos[0].cpu().numpy()  # [3]
                phase_int   = int(policy.phases[0].item())
                obs, rewards, terms, truncs, infos = collector.step(actions)
                terminated = bool(terms[0].item())
                truncated  = bool(truncs[0].item())
                raw_success = infos.get("success", False)
                success = bool(
                    raw_success[0].item()
                    if isinstance(raw_success, torch.Tensor)
                    else raw_success
                )
                done = terminated or truncated

                # 2. extract RGB → send to encoder server
                collector.extract_rgb(obs, frame_buf)
                frame_queue.put((worker_id, frame_buf.cpu().numpy()[0]))  # [3,H,W]

                # 3. wait for latent
                latent_np = result_queue.get()   # [16,8,8] float16

                # 4. write
                writer.append_frame(
                    env_id=0,
                    latent=latent_np,
                    action=actions[0].cpu().numpy(),
                    reward=float(rewards[0]),
                    terminated=terminated,
                    truncated=truncated,
                    success=success,
                    ee_pos=ee_pos_np,
                    cube_pos=cube_pos_np,
                    phase=phase_int,
                )

                if done:
                    policy.reset_done_envs(torch.tensor([True], device=fb_device))
                    writer.finalize_episode(
                        env_id=0, task=cfg.task, seed=cfg.seed,
                        completed=bool(terms[0].item()),
                    )
                    completed += 1
                    if completed % 100 == 0:
                        print(f"[Worker {worker_id:02d}] {completed}/{cfg.max_episodes} episodes",
                              flush=True)
                    writer.ensure_episode(0)
                    obs, _ = collector.reset()

    # signal encoder server that this worker is done
    frame_queue.put(_SENTINEL)
    print(f"[Worker {worker_id:02d}] finished  {completed} episodes written to {cfg.hdf5_path}",
          flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Launcher
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--num_workers",          type=int,   default=8)
    p.add_argument("--episodes_per_worker",  type=int,   default=1250)
    p.add_argument("--max_steps",            type=int,   default=400)
    p.add_argument("--shard_dir",            type=str,   default="trajectories_shards")
    p.add_argument("--task",                 type=str,   default="PickCube-v1")
    p.add_argument("--cosmos_ckpt",          type=str,
                   default="pretrained_ckpts/Cosmos-Tokenizer-CI16x16")
    p.add_argument("--noise_scale",          type=float, default=0.13)
    p.add_argument("--base_seed",            type=int,   default=42)
    p.add_argument("--sim_backend",          type=str,   default="cpu",
                   choices=["gpu", "cpu"],
                   help="Physics backend. Each worker always uses num_envs=1.")
    p.add_argument("--robot_init_qpos_noise", type=float, default=0.10,
                   help="Std (rad) of joint-angle noise applied to robot at episode start.")
    p.add_argument("--cube_spawn_half_size",  type=float, default=0.15,
                   help="Half-side (m) of the XY region where cube and goal are spawned.")
    return p.parse_args()


def main():
    args = parse_args()
    mp.set_start_method("spawn", force=True)

    shard_dir = Path(args.shard_dir)
    shard_dir.mkdir(exist_ok=True)

    total = args.num_workers * args.episodes_per_worker
    print(f"Collecting {total} episodes  "
          f"({args.num_workers} workers × {args.episodes_per_worker} each)")
    print(f"Shards → {shard_dir.resolve()}\n")

    # verify Cosmos weights before spawning anything
    ensure_cosmos_weights(args.cosmos_ckpt)

    # one queue for frames (workers → encoder), one result queue per worker
    frame_queue   = mp.Queue(maxsize=args.num_workers * 4)
    result_queues = [mp.Queue(maxsize=8) for _ in range(args.num_workers)]

    # encoder server
    enc_proc = mp.Process(
        target=encoder_server,
        args=(args.cosmos_ckpt, frame_queue, result_queues, args.num_workers),
        daemon=True,
    )
    enc_proc.start()
    print(f"[Launcher] encoder server PID={enc_proc.pid}")

    # workers
    worker_procs = []
    for i in range(args.num_workers):
        cfg = IngestConfig(
            task=args.task,
            num_envs=1,
            max_episodes=args.episodes_per_worker,
            max_steps=args.max_steps,
            hdf5_path=str(shard_dir / f"shard_{i:02d}.h5"),
            cosmos_ckpt=args.cosmos_ckpt,
            seed=args.base_seed + i,
            async_writer=False,
            noise_scale=args.noise_scale,
            sim_backend=args.sim_backend,
            robot_init_qpos_noise=args.robot_init_qpos_noise,
            cube_spawn_half_size=args.cube_spawn_half_size,
        )
        p = mp.Process(target=worker, args=(i, cfg, frame_queue, result_queues[i]))
        p.start()
        worker_procs.append(p)
        print(f"[Launcher] worker {i:02d} PID={p.pid}  seed={cfg.seed}  → {cfg.hdf5_path}")

    print(f"\n[Launcher] all processes running...\n")
    t0 = time.perf_counter()

    # wait for workers
    failed = []
    for i, p in enumerate(worker_procs):
        p.join()
        if p.exitcode != 0:
            failed.append(i)
            print(f"[Launcher] worker {i:02d} FAILED (exit={p.exitcode})")
        else:
            print(f"[Launcher] worker {i:02d} done")

    enc_proc.join(timeout=30)
    if enc_proc.is_alive():
        enc_proc.terminate()

    wall = time.perf_counter() - t0
    print(f"\nWall time: {wall / 60:.1f} min")

    if failed:
        print(f"\n{len(failed)} workers failed: {failed}")
        sys.exit(1)
    else:
        print(f"\nAll workers complete. Merge with:")
        print(f"  python scripts/merge_shards.py "
              f"--shard_dir {shard_dir} --output trajectories_20k.h5")


if __name__ == "__main__":
    main()
