"""DiT + Conditional Flow Matching training loop.

Features:
  - torch.amp mixed-precision (bfloat16 training per CLAUDE.md)
  - Gradient clipping (max_norm=1.0)
  - Cosine annealing LR scheduler with warmup
  - WandB logging (loss, lr, grad_norm, GPU memory)
  - Validation GIF every N steps: 8-step Heun rollout decoded to pixel space
  - Checkpoint saving

Usage:
    python -m training.train --hdf5 trajectories_10k.h5 --epochs 100

On Colab A100:
    See COLAB_TRAINING.md for setup instructions.
"""

from __future__ import annotations

import argparse
import math
import os
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import record_function
from torch.utils.data import DataLoader

from data.dataset import TrajectoryDataset
from models.dit import DiTSmall, IN_CHANNELS, LATENT_H, LATENT_W


# ---------------------------------------------------------------------------
# Heun ODE sampler (2nd-order, for validation quality)
# ---------------------------------------------------------------------------
@torch.no_grad()
def sample_heun(
    model: nn.Module,
    cond: torch.Tensor,
    ctx_latents: torch.Tensor | None = None,
    num_steps: int = 8,
) -> torch.Tensor:
    """Heun's method (improved Euler) ODE sampler.

    Passes ctx_latents through to model.forward at every ODE step, which
    is equivalent to (and consistent with) the KV-cached inference path.
    """
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        B = cond.shape[0]
        x = torch.randn(B, IN_CHANNELS, LATENT_H, LATENT_W, device=cond.device)
        dt = 1.0 / num_steps

        for i in range(num_steps):
            t = torch.full((B,), i * dt, device=x.device)
            v1 = model(x, t, cond, ctx_latents=ctx_latents)
            x_euler = x + dt * v1

            if i < num_steps - 1:
                t_next = torch.full((B,), (i + 1) * dt, device=x.device)
                v2 = model(x_euler, t_next, cond, ctx_latents=ctx_latents)
                x = x + dt * 0.5 * (v1 + v2)
            else:
                x = x_euler

        return x


# ---------------------------------------------------------------------------
# Cosine schedule with linear warmup
# ---------------------------------------------------------------------------
def cosine_warmup_schedule(optimizer, warmup_steps: int, total_steps: int):
    """Returns a LambdaLR with linear warmup then cosine decay to 0."""

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Validation GIF generation
# ---------------------------------------------------------------------------
def make_validation_gif(
    model: nn.Module,
    val_loader: DataLoader,
    step: int,
    output_dir: str,
    decoder=None,
) -> str | None:
    """Generate an 8-step Heun rollout, save as GIF. Returns path or None."""
    try:
        import imageio.v3 as iio
    except ImportError:
        print("imageio not installed, skipping GIF generation")
        return None

    model.eval()
    batch = next(iter(val_loader))
    x_1  = batch["x_1"][:4].cuda()
    cond = batch["cond"][:4].cuda()
    ctx  = batch.get("ctx_latents")
    if ctx is not None:
        ctx = ctx[:4].cuda()

    # Generate predictions
    x_pred = sample_heun(model, cond, ctx_latents=ctx, num_steps=8)

    # Stack ground-truth and predicted for comparison
    # Each is [4, 16, 8, 8] — take first 3 channels as pseudo-RGB
    frames = []
    for pair_idx in range(min(4, x_1.shape[0])):
        gt = x_1[pair_idx, :3].float().cpu()
        pred = x_pred[pair_idx, :3].float().cpu()

        # Normalize to [0, 255]
        for tensor in [gt, pred]:
            tensor.sub_(tensor.min()).div_(tensor.max() - tensor.min() + 1e-5)

        # Side-by-side: [3, 8, 16] -> [8, 16, 3]
        combined = torch.cat([gt, pred], dim=2)  # [3, 8, 16]
        frame = (combined.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        # Upscale for visibility
        frame = np.repeat(np.repeat(frame, 16, axis=0), 16, axis=1)
        frames.append(frame)

    gif_path = os.path.join(output_dir, f"val_step_{step:06d}.gif")
    iio.imwrite(gif_path, frames, duration=500, loop=0)
    model.train()
    return gif_path


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def train(args):
    # ---- Setup ----
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda")

    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.gif_dir, exist_ok=True)

    # ---- WandB ----
    run = None
    if not args.no_wandb:
        import wandb

        run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
        )

    # ---- Data ----
    dataset = TrajectoryDataset(args.hdf5, ctx_frames=args.ctx_frames)
    print(f"Dataset: {len(dataset)} samples from {args.hdf5}  ctx_frames={args.ctx_frames}")

    # 90/10 train/val split
    val_size = max(1, len(dataset) // 10)
    train_size = len(dataset) - val_size
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # ---- Model ----
    model = DiTSmall().to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # ---- Optimizer + Scheduler ----
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )

    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = min(args.warmup_steps, total_steps // 10)
    scheduler = cosine_warmup_schedule(optimizer, warmup_steps, total_steps)

    # ---- Mixed precision scaler (bfloat16 per CLAUDE.md) ----
    # bfloat16 doesn't need GradScaler (no underflow risk), but we use autocast
    use_bf16 = torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(not use_bf16))
    print(f"AMP dtype: {amp_dtype}, GradScaler: {not use_bf16}")

    # ---- Resume from checkpoint ----
    start_epoch = 0
    global_step = 0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        if not use_bf16 and "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt["global_step"]
        print(f"Resumed from {args.resume} (epoch {start_epoch}, step {global_step})")

    # ---- Training ----
    model.train()
    best_val_loss = float("inf")

    for epoch in range(start_epoch, args.epochs):
        epoch_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            x_1      = batch["x_1"].to(device, non_blocking=True)
            cond     = batch["cond"].to(device, non_blocking=True)      # [B, 4]
            cube_pos = batch["cube_pos"].to(device, non_blocking=True)  # [B, 3]
            ctx      = batch.get("ctx_latents")
            if ctx is not None:
                ctx = ctx.to(device, non_blocking=True)                 # [B, n_ctx, 16, 8, 8]
            B = x_1.shape[0]

            # --- CFM forward (inline for efficiency) ---
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                with record_function("CFM_forward"):
                    t = torch.rand(B, device=device)
                    x_0 = torch.randn_like(x_1)
                    t_exp = t.view(B, 1, 1, 1)
                    x_t = (1 - t_exp) * x_0 + t_exp * x_1
                    v_target = x_1 - x_0
                    v_pred, cube_pos_pred = model(x_t, t, cond,
                                                  ctx_latents=ctx, return_aux=True)
                    loss_cfm = F.mse_loss(v_pred, v_target)
                    loss_aux = F.mse_loss(cube_pos_pred, cube_pos)
                    loss = loss_cfm + 0.1 * loss_aux

            # --- Backward ---
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()

            # Gradient clipping
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=args.grad_clip
            )

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            loss_val = loss.item()
            epoch_loss += loss_val
            num_batches += 1
            global_step += 1

            # --- Logging ---
            if global_step % args.log_every == 0:
                lr_current   = optimizer.param_groups[0]["lr"]
                mem_mb       = torch.cuda.max_memory_allocated() / (1024 * 1024)
                loss_cfm_val = loss_cfm.item()
                loss_aux_val = loss_aux.item()
                print(
                    f"[step {global_step:>6d} | epoch {epoch:>3d}] "
                    f"loss={loss_val:.4f}  cfm={loss_cfm_val:.4f}  "
                    f"aux={loss_aux_val:.4f}  grad_norm={grad_norm:.3f}  "
                    f"lr={lr_current:.2e}  mem={mem_mb:.0f}MB"
                )
                if run is not None:
                    run.log(
                        {
                            "train/loss":      loss_val,
                            "train/loss_cfm":  loss_cfm_val,
                            "train/loss_aux":  loss_aux_val,
                            "train/grad_norm": float(grad_norm),
                            "train/lr":        lr_current,
                            "train/epoch":     epoch,
                            "gpu/mem_peak_mb": mem_mb,
                        },
                        step=global_step,
                    )

            # --- Validation GIF ---
            if global_step % args.val_every == 0:
                model.eval()

                # Compute val loss
                val_losses = []
                with torch.no_grad(), torch.amp.autocast("cuda", dtype=amp_dtype):
                    for i, vbatch in enumerate(val_loader):
                        if i >= args.val_batches:
                            break
                        vx1   = vbatch["x_1"].to(device, non_blocking=True)
                        vcond = vbatch["cond"].to(device, non_blocking=True)
                        vctx  = vbatch.get("ctx_latents")
                        if vctx is not None:
                            vctx = vctx.to(device, non_blocking=True)
                        vB = vx1.shape[0]
                        vt = torch.rand(vB, device=device)
                        vx0 = torch.randn_like(vx1)
                        vt_exp = vt.view(vB, 1, 1, 1)
                        vx_t = (1 - vt_exp) * vx0 + vt_exp * vx1
                        vv_target = vx1 - vx0
                        vv_pred = model(vx_t, vt, vcond, ctx_latents=vctx)
                        val_losses.append(F.mse_loss(vv_pred, vv_target).item())

                val_loss = sum(val_losses) / len(val_losses) if val_losses else 0.0
                print(f"  >> val_loss={val_loss:.4f}")

                if run is not None:
                    log_dict = {"val/loss": val_loss}

                    gif_path = make_validation_gif(
                        model, val_loader, global_step, args.gif_dir
                    )
                    if gif_path is not None:
                        import wandb
                        log_dict["val/rollout_gif"] = wandb.Video(gif_path, fps=2, format="gif")

                    run.log(log_dict, step=global_step)

                # Save best
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(
                        model, optimizer, scheduler, scaler, epoch, global_step,
                        os.path.join(args.ckpt_dir, "best.pt"),
                    )
                    print(f"  >> new best val_loss={val_loss:.4f}")

                model.train()

        # --- End of epoch ---
        avg_loss = epoch_loss / max(1, num_batches)
        print(f"Epoch {epoch} done. avg_loss={avg_loss:.4f}")

        # Periodic checkpoint
        if (epoch + 1) % args.save_every_epochs == 0:
            save_checkpoint(
                model, optimizer, scheduler, scaler, epoch, global_step,
                os.path.join(args.ckpt_dir, f"epoch_{epoch:04d}.pt"),
            )

    # Final checkpoint
    save_checkpoint(
        model, optimizer, scheduler, scaler, args.epochs - 1, global_step,
        os.path.join(args.ckpt_dir, "final.pt"),
    )

    if run is not None:
        run.finish()

    print(f"Training complete. Best val_loss={best_val_loss:.4f}")


def save_checkpoint(model, optimizer, scheduler, scaler, epoch, global_step, path):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
        },
        path,
    )
    print(f"  >> saved {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="DiT + CFM training")
    # Data
    p.add_argument("--hdf5", type=str, default="trajectories_10k_v2.h5")
    p.add_argument("--ctx_frames", type=int, default=4,
                   help="Context frames fed to the model during training (0 disables)")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=4)
    # Training
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--warmup_steps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    # Logging
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--val_every", type=int, default=1000)
    p.add_argument("--val_batches", type=int, default=10)
    p.add_argument("--no_wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="optiworld-dit")
    p.add_argument("--wandb_run_name", type=str, default=None)
    # Checkpoints
    p.add_argument("--ckpt_dir", type=str, default="checkpoints")
    p.add_argument("--gif_dir", type=str, default="val_gifs")
    p.add_argument("--save_every_epochs", type=int, default=10)
    p.add_argument("--resume", type=str, default=None)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
