"""Visualize predicted frames from a trained world model.

Loads a DiT checkpoint, samples predicted latents, decodes them to RGB,
and saves side-by-side comparison GIFs: [context frames... | prediction | ground truth]

Usage:
    python scripts/visualize_predictions.py \\
        --ckpt checkpoints/model.pt \\
        --data trajectories.h5 \\
        --cosmos_ckpt pretrained_ckpts/Cosmos-Tokenizer-CI16x16 \\
        --n_ctx 4 --num_steps 16 --out predictions_16steps.gif
"""

from __future__ import annotations

import argparse
import os

import h5py
import numpy as np
import torch

from data.ingest import CosmosLatentDecoder
from inference.solver import sample_heun_cached
from models.dit import DiTSmall


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---- Load model ----
    print(f"Loading model from {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = DiTSmall().to(device)
    # Strip torch.compile _orig_mod. prefix if present
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

    # ---- Load data ----
    print(f"Loading latents from {args.data}")
    with h5py.File(args.data, "r") as f:
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
                print(f"Using episode: {episode_name}")
                break

            tries += 1
            episode_idx = np.random.randint(len(episode_names))

        if tries >= max_tries:
            print(f"Could not find a valid episode after {max_tries} tries. Exiting.")
            return

        actions = episode_group["actions"][:]  # [T, 4]

        print(f"Episode shape: latents={latents.shape}, actions={actions.shape}")

        # Pick a random starting point with enough context
        max_start = latents.shape[0] - 1
        if max_start < args.n_ctx:
            print(f"Episode too short (need at least {args.n_ctx+1} frames), skipping")
            return

        start_idx = np.random.randint(args.n_ctx, max_start)
        ctx_latents = latents[start_idx - args.n_ctx:start_idx]  # [n_ctx, 16, 8, 8]
        pred_action = actions[start_idx]  # [4] — action to predict next frame for
        gt_latent = latents[start_idx + 1]  # [16, 8, 8] — ground truth next frame

    # ---- Inference ----
    print(f"Running inference with {args.n_ctx} context frames...")
    with torch.no_grad():
        ctx_latents_t = torch.from_numpy(ctx_latents).float().to(device).unsqueeze(0)  # [1, n_ctx, 16, 8, 8]
        pred_action_t = torch.from_numpy(pred_action).float().to(device).unsqueeze(0)  # [1, 4]

        print(f"  Context latents: min={ctx_latents_t.min():.4f}, max={ctx_latents_t.max():.4f}, mean={ctx_latents_t.mean():.4f}")
        print(f"  Action: {pred_action_t.squeeze().cpu().numpy()}")

        # Predict next latent: given context frames + action, predict next frame
        pred_latent = sample_heun_cached(
            model,
            ctx_latents_t,
            pred_action_t,
            num_steps=args.num_steps,
        )  # [1, 16, 8, 8]

        print(f"  Predicted latent: min={pred_latent.min():.4f}, max={pred_latent.max():.4f}, mean={pred_latent.mean():.4f}")
        print(f"  Contains NaN: {torch.isnan(pred_latent).any().item()}")

        gt_latent_t = torch.from_numpy(gt_latent).float().to(device).unsqueeze(0)  # [1, 16, 8, 8]

    print("Inference done.")

    # ---- Decode all frames to RGB ----
    print("Decoding latents to RGB...")
    with torch.no_grad():
        # Decode context frames
        ctx_frames = []
        for i in range(ctx_latents.shape[0]):
            latent = torch.from_numpy(ctx_latents[i]).float().to(device).unsqueeze(0)  # [1, 16, 8, 8]
            rgb = decoder.decode(latent)[0].cpu()  # [3, 128, 128]
            # Handle NaN and convert to uint8
            rgb_np = rgb.permute(1, 2, 0).numpy()
            rgb_np = np.nan_to_num(rgb_np, nan=0.0)  # Replace NaN with 0
            rgb_np = np.clip(rgb_np, 0, 1)  # Ensure [0, 1] range
            ctx_frames.append((rgb_np * 255).astype(np.uint8))

        # Decode predicted frame
        pred_rgb = decoder.decode(pred_latent)[0].cpu()  # [3, 128, 128]
        pred_np = pred_rgb.permute(1, 2, 0).numpy()
        pred_np = np.nan_to_num(pred_np, nan=0.0)
        pred_np = np.clip(pred_np, 0, 1)
        pred_frame = (pred_np * 255).astype(np.uint8)

        # Decode ground truth frame
        gt_rgb = decoder.decode(gt_latent_t)[0].cpu()  # [3, 128, 128]
        gt_np = gt_rgb.permute(1, 2, 0).numpy()
        gt_np = np.nan_to_num(gt_np, nan=0.0)
        gt_np = np.clip(gt_np, 0, 1)
        gt_frame = (gt_np * 255).astype(np.uint8)

    print("Decoding done.")
    print(f"Pred frame stats: min={pred_frame.min()}, max={pred_frame.max()}, mean={pred_frame.mean():.1f}")
    print(f"GT frame stats: min={gt_frame.min()}, max={gt_frame.max()}, mean={gt_frame.mean():.1f}")

    # ---- Create comparison frames ----
    print("Creating comparison visualization...")
    frames = []
    for i, ctx_frame in enumerate(ctx_frames):
        # Stack [ctx] side-by-side with prediction/gt
        combined = np.concatenate([ctx_frame, pred_frame, gt_frame], axis=1)  # [128, 384, 3]
        frames.append(combined)
        if i == 0:
            print(f"Frame shape: {combined.shape}")

    # ---- Save GIF ----
    print(f"Saving GIF to {args.out}")
    try:
        import imageio.v3 as iio

        iio.imwrite(args.out, frames, duration=300, loop=0)
        print(f"GIF saved: {args.out}")
    except ImportError:
        print("imageio not installed, saving as PNG sequence instead")
        os.makedirs(args.out, exist_ok=True)
        for i, frame in enumerate(frames):
            iio.imwrite(os.path.join(args.out, f"frame_{i:03d}.png"), frame)
        print(f"PNG frames saved to {args.out}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize predictions from a trained world model"
    )
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
        "--n_ctx",
        type=int,
        default=4,
        help="Number of context frames to condition on",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=8,
        help="Number of ODE solver steps for denoising",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="predictions.gif",
        help="Output GIF path",
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
