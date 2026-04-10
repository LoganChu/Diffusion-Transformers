"""Visualize predicted frames from a trained world model.

Loads a DiT checkpoint, samples predicted latents, decodes them to RGB,
and saves side-by-side comparison GIFs: [context frames... | prediction | ground truth]

Usage:
    python scripts/visualize_predictions.py \\
        --ckpt checkpoints/model.pt \\
        --data trajectories.h5 \\
        --cosmos_ckpt pretrained_ckpts/Cosmos-Tokenizer-CI16x16 \\
        --n_ctx 4 --out predictions.gif
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
        # Pick a random episode
        episode_names = list(f.keys())
        episode_idx = np.random.randint(len(episode_names))
        episode_name = episode_names[episode_idx]
        print(f"Using episode: {episode_name}")

        episode_group = f[episode_name]
        latents = episode_group["latents"][:]  # [T, 16, 8, 8]
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

        # Predict next latent: given context frames + action, predict next frame
        pred_latent = sample_heun_cached(
            model,
            ctx_latents_t,
            pred_action_t,
            num_steps=8,
        )  # [1, 16, 8, 8]

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
            ctx_frames.append((rgb.permute(1, 2, 0).numpy() * 255).astype(np.uint8))

        # Decode predicted frame
        pred_rgb = decoder.decode(pred_latent)[0].cpu()  # [3, 128, 128]
        pred_frame = (pred_rgb.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        # Decode ground truth frame
        gt_rgb = decoder.decode(gt_latent_t)[0].cpu()  # [3, 128, 128]
        gt_frame = (gt_rgb.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    print("Decoding done.")

    # ---- Create comparison frames ----
    print("Creating comparison visualization...")
    frames = []
    for ctx_frame in ctx_frames:
        # Stack [ctx] side-by-side with prediction/gt
        combined = np.concatenate([ctx_frame, pred_frame, gt_frame], axis=1)  # [128, 384, 3]
        frames.append(combined)

    # Add a final frame showing [pred | gt] more clearly
    combined_final = np.concatenate([pred_frame, gt_frame], axis=1)  # [128, 256, 3]
    frames.append(combined_final)

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
