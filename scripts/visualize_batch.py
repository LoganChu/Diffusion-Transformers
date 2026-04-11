"""Generate multiple prediction visualizations from different episodes.

Usage:
    python scripts/visualize_batch.py \\
        --ckpt offline_best.pt \\
        --data trajectories_10k.h5 \\
        --cosmos_ckpt pretrained_ckpts/Cosmos-Tokenizer-CI16x16 \\
        --n_pred 10 --n_ctx 4 --out_dir predictions_batch
"""

from __future__ import annotations

import argparse
import os
import shutil

import h5py
import numpy as np
import torch

from data.ingest import CosmosLatentDecoder
from inference.solver import sample_heun_cached
from models.dit import DiTSmall


def generate_prediction(
    model: torch.nn.Module,
    decoder: CosmosLatentDecoder,
    h5_path: str,
    n_ctx: int,
    device: torch.device,
    pred_idx: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a single prediction and return pred_frame, gt_frame, ctx_frames."""
    with h5py.File(h5_path, "r") as f:
        # Pick a random episode (skip if it contains NaN)
        episode_names = list(f.keys())
        episode_idx = np.random.randint(len(episode_names))
        max_tries = 10
        tries = 0

        while tries < max_tries:
            episode_name = episode_names[episode_idx]
            episode_group = f[episode_name]
            latents = episode_group["latents"][:]  # [T, 16, 8, 8]

            # Check if this episode has valid data
            if not np.isnan(latents).any():
                break

            tries += 1
            episode_idx = np.random.randint(len(episode_names))

        if tries >= max_tries:
            print(f"Pred {pred_idx}: Could not find valid episode")
            return None, None, None

        actions = episode_group["actions"][:]  # [T, 4]

        # Pick a random starting point with enough context
        max_start = latents.shape[0] - 1
        if max_start < n_ctx:
            print(f"Pred {pred_idx}: Episode too short")
            return None, None, None

        start_idx = np.random.randint(n_ctx, max_start)
        ctx_latents = latents[start_idx - n_ctx : start_idx]  # [n_ctx, 16, 8, 8]
        pred_action = actions[start_idx]  # [4]
        gt_latent = latents[start_idx + 1]  # [16, 8, 8]

    # ---- Inference ----
    with torch.no_grad():
        ctx_latents_t = torch.from_numpy(ctx_latents).float().to(device).unsqueeze(0)
        pred_action_t = torch.from_numpy(pred_action).float().to(device).unsqueeze(0)

        pred_latent = sample_heun_cached(
            model,
            ctx_latents_t,
            pred_action_t,
            num_steps=8,
        )  # [1, 16, 8, 8]

        gt_latent_t = torch.from_numpy(gt_latent).float().to(device).unsqueeze(0)

    # ---- Decode all frames to RGB ----
    with torch.no_grad():
        # Decode context frames
        ctx_frames = []
        for i in range(ctx_latents.shape[0]):
            latent = torch.from_numpy(ctx_latents[i]).float().to(device).unsqueeze(0)
            rgb = decoder.decode(latent)[0].cpu()
            rgb_np = rgb.permute(1, 2, 0).numpy()
            rgb_np = np.nan_to_num(rgb_np, nan=0.0)
            rgb_np = np.clip(rgb_np, 0, 1)
            ctx_frames.append((rgb_np * 255).astype(np.uint8))

        # Decode predicted frame
        pred_rgb = decoder.decode(pred_latent)[0].cpu()
        pred_np = pred_rgb.permute(1, 2, 0).numpy()
        pred_np = np.nan_to_num(pred_np, nan=0.0)
        pred_np = np.clip(pred_np, 0, 1)
        pred_frame = (pred_np * 255).astype(np.uint8)

        # Decode ground truth frame
        gt_rgb = decoder.decode(gt_latent_t)[0].cpu()
        gt_np = gt_rgb.permute(1, 2, 0).numpy()
        gt_np = np.nan_to_num(gt_np, nan=0.0)
        gt_np = np.clip(gt_np, 0, 1)
        gt_frame = (gt_np * 255).astype(np.uint8)

    return pred_frame, gt_frame, ctx_frames


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---- Load model ----
    print(f"Loading model from {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = DiTSmall().to(device)
    state_dict = ckpt["model"]
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    print("Model loaded.")

    # ---- Load VAE decoder ----
    print(f"Loading VAE decoder from {args.cosmos_ckpt}")
    decoder = CosmosLatentDecoder(args.cosmos_ckpt)
    print("VAE decoder loaded.")

    # ---- Create output directory ----
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Saving predictions to {args.out_dir}")

    # ---- Generate multiple predictions ----
    successful = 0
    for pred_idx in range(args.n_pred):
        print(f"\n[{pred_idx+1}/{args.n_pred}] Generating prediction...")

        pred_frame, gt_frame, ctx_frames = generate_prediction(
            model, decoder, args.data, args.n_ctx, device, pred_idx
        )

        if pred_frame is None:
            print(f"  Skipped (no valid episode)")
            continue

        # Create comparison frames
        frames = []
        for ctx_frame in ctx_frames:
            combined = np.concatenate([ctx_frame, pred_frame, gt_frame], axis=1)
            frames.append(combined)

        # Save GIF
        out_path = os.path.join(args.out_dir, f"prediction_{successful:02d}.gif")
        try:
            import imageio.v3 as iio

            iio.imwrite(out_path, frames, duration=300, loop=0)
            print(f"  [OK] Saved: {out_path}")
            successful += 1
        except ImportError:
            print("  imageio not installed, skipping GIF save")

    print(f"\nGenerated {successful}/{args.n_pred} predictions")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate multiple prediction visualizations")
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to trained model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to HDF5 trajectory dataset",
    )
    parser.add_argument(
        "--cosmos_ckpt",
        type=str,
        default="pretrained_ckpts/Cosmos-Tokenizer-CI16x16",
        help="Path to Cosmos tokenizer checkpoint directory",
    )
    parser.add_argument(
        "--n_pred",
        type=int,
        default=10,
        help="Number of predictions to generate",
    )
    parser.add_argument(
        "--n_ctx",
        type=int,
        default=4,
        help="Number of context frames to condition on",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="predictions_batch",
        help="Output directory for GIFs",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    main(args)
