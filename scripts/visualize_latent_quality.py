"""Render two episodes and annotate frames where the Cosmos encoder produces
inf/NaN latents. Saves two MP4s:
  outputs/videos/latent_bad_seed{seed}.mp4   — episode with bad frames highlighted red
  outputs/videos/latent_clean_seed{seed}.mp4 — clean episode (no bad frames)

Usage:
    conda run -n ml_env python scripts/visualize_latent_quality.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import imageio
import numpy as np
import torch
import gymnasium as gym
import mani_skill.envs  # noqa: F401

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.ingest import (
    CosmosLatentEncoder,
    GuidedPolicy,
    IngestConfig,
    ManiSkillCollector,
    OBS_H,
    OBS_W,
    ensure_cosmos_weights,
)

COSMOS_CKPT  = "pretrained_ckpts/Cosmos-Tokenizer-CI16x16"
VIDEO_DIR    = Path("outputs/videos")
MAX_STEPS    = 400
FPS          = 20
TASK         = "PickCube-v1"


def red_border(frame: np.ndarray, thickness: int = 8) -> np.ndarray:
    """Paint a red border around a uint8 HWC frame."""
    f = frame.copy()
    f[:thickness, :] = [255, 0, 0]
    f[-thickness:, :] = [255, 0, 0]
    f[:, :thickness] = [255, 0, 0]
    f[:, -thickness:] = [255, 0, 0]
    return f


def label_frame(frame: np.ndarray, text: str, color=(255, 255, 255)) -> np.ndarray:
    """Stamp a text label in the top-left corner (no cv2 dependency)."""
    # Simple pixel font substitute: just tint a strip at the top
    f = frame.copy()
    strip_h = 24
    if text == "INF/NAN LATENT":
        f[:strip_h, :] = [200, 0, 0]
    else:
        f[:strip_h, :] = [0, 160, 0]
    return f


def render_annotated_episode(
    cfg: IngestConfig,
    encoder: CosmosLatentEncoder,
    seed: int,
    label: str,
) -> tuple[list[np.ndarray], int]:
    """Roll out one episode, annotate frames where latent is bad.
    Returns (frames, n_bad_frames).
    """
    frame_buf = torch.empty(1, 3, OBS_H, OBS_W, dtype=torch.float16, device="cuda")
    frames_out = []
    n_bad = 0

    with ManiSkillCollector(cfg) as collector:
        policy = GuidedPolicy(
            env=collector.env,
            num_envs=1,
            noise_scale=cfg.noise_scale,
            device="cpu",
        )
        obs, _ = collector.reset()

        for _step in range(MAX_STEPS):
            action = policy()
            obs, _, terms, truncs, _ = collector.step(action)
            done = bool(terms[0].item()) or bool(truncs[0].item())

            # Get RGB frame for video
            raw = collector.env.render()
            if hasattr(raw, "cpu"):
                raw = raw.cpu().numpy()
            if raw.ndim == 4:
                raw = raw[0]
            frame_rgb = raw  # [H, W, 3] uint8

            # Encode and check for inf/NaN
            collector.extract_rgb(obs, frame_buf)
            with torch.no_grad():
                latent = encoder.encode(frame_buf)
                encoder.sync()
            latent_np = latent.cpu().float().numpy()
            is_bad = not np.isfinite(latent_np).all()

            if is_bad:
                n_bad += 1
                annotated = red_border(frame_rgb)
                annotated = label_frame(annotated, "INF/NAN LATENT")
            else:
                annotated = label_frame(frame_rgb, "OK")

            frames_out.append(annotated)

            if done:
                break

    return frames_out, n_bad


def find_bad_and_clean_seeds(n_trials: int = 20) -> tuple[int | None, int | None]:
    """Quick scan: run short episodes to find a seed with bad latents and one without."""
    ensure_cosmos_weights(COSMOS_CKPT)
    encoder = CosmosLatentEncoder(COSMOS_CKPT)
    frame_buf = torch.empty(1, 3, OBS_H, OBS_W, dtype=torch.float16, device="cuda")

    cfg = IngestConfig(
        task=TASK, num_envs=1, max_episodes=1, max_steps=MAX_STEPS,
        hdf5_path="/dev/null", cosmos_ckpt=COSMOS_CKPT,
        seed=42, sim_backend="cpu",
    )

    bad_seed = None
    clean_seed = None

    for seed in range(n_trials):
        if bad_seed and clean_seed:
            break
        cfg.seed = seed
        found_bad = False
        with ManiSkillCollector(cfg) as collector:
            policy = GuidedPolicy(env=collector.env, num_envs=1,
                                  noise_scale=0.13, device="cpu")
            obs, _ = collector.reset()
            for _step in range(MAX_STEPS):
                action = policy()
                obs, _, terms, truncs, _ = collector.step(action)
                collector.extract_rgb(obs, frame_buf)
                with torch.no_grad():
                    latent = encoder.encode(frame_buf)
                    encoder.sync()
                latent_np = latent.cpu().float().numpy()
                if not np.isfinite(latent_np).all():
                    found_bad = True
                if bool(terms[0].item()) or bool(truncs[0].item()):
                    break

        if found_bad and bad_seed is None:
            bad_seed = seed
            print(f"Found bad seed: {seed}")
        elif not found_bad and clean_seed is None:
            clean_seed = seed
            print(f"Found clean seed: {seed}")

    return bad_seed, clean_seed


def main():
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    ensure_cosmos_weights(COSMOS_CKPT)
    encoder = CosmosLatentEncoder(COSMOS_CKPT)

    print("Scanning for a bad and clean seed...")
    bad_seed, clean_seed = find_bad_and_clean_seeds(n_trials=30)

    if bad_seed is None:
        print("No bad seed found in 30 trials — latents may all be clean with current encoder.")
        bad_seed = 0
    if clean_seed is None:
        print("No clean seed found — all episodes have bad latents.")
        clean_seed = 1

    cfg_base = IngestConfig(
        task=TASK, num_envs=1, max_episodes=1, max_steps=MAX_STEPS,
        hdf5_path="/dev/null", cosmos_ckpt=COSMOS_CKPT,
        seed=42, sim_backend="cpu",
        noise_scale=0.13,
    )

    for seed, label, tag in [(bad_seed, "bad", "bad"), (clean_seed, "clean", "clean")]:
        print(f"\nRendering {label} episode (seed={seed})...")
        cfg_base.seed = seed
        frames, n_bad = render_annotated_episode(cfg_base, encoder, seed, label)
        out_path = VIDEO_DIR / f"latent_{tag}_seed{seed}.mp4"
        imageio.mimsave(str(out_path), frames, fps=FPS)
        print(f"  {len(frames)} frames, {n_bad} bad latent frames → {out_path}")

    print("\nDone. Green strip = clean latent, Red border+strip = inf/NaN latent.")


if __name__ == "__main__":
    main()
