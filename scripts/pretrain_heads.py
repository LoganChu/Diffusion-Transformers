"""Standalone head pretraining script.

Loads an offline checkpoint, seeds a replay buffer from the offline HDF5,
pre-trains the reward / done / value heads with the backbone frozen, then
saves a new checkpoint ready for online training.

The backbone weights are NOT modified — only the three task head MLPs change.
Run diagnose_cem.py afterward to verify reward_value score discriminability.

Usage
-----
    python -m scripts.pretrain_heads \
        --checkpoint offline_best.pt \
        --hdf5       trajectories_10k.h5

    # Custom output path and more steps:
    python -m scripts.pretrain_heads \
        --checkpoint offline_best.pt \
        --hdf5       trajectories_10k.h5 \
        --output     offline_best_heads.pt \
        --steps      3000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from data.replay        import ReplayBuffer
from models.dit         import ACTION_DIM, DiTSmall
from training.train_online import pretrain_heads


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _head_norm(model) -> dict[str, float]:
    """Return the L2 norm of the final-layer weight for each task head.

    A norm near zero means the head is still at its zero-initialization.
    A non-zero norm confirms the head has been trained.
    """
    return {
        "reward": model.reward_head.mlp[-1].weight.norm().item(),
        "done":   model.done_head.mlp[-1].weight.norm().item(),
        "value":  model.value_head.mlp[-1].weight.norm().item(),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Load checkpoint ----
    print(f"\nLoading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    if "episode" in ckpt:
        print(
            f"  WARNING: checkpoint contains 'episode'={ckpt['episode']} — "
            "this looks like an online checkpoint, not an offline one.\n"
            "  Head pretraining is most useful on offline checkpoints whose "
            "task heads are still zero-initialized.\n"
            "  Proceeding anyway."
        )
    else:
        ep = ckpt.get("epoch", "?")
        print(f"  Offline checkpoint confirmed  (epoch={ep})")

    model = DiTSmall().to(device)
    sd = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
    model.load_state_dict(sd)
    print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")

    # ---- Head norms before training ----
    norms_before = _head_norm(model)
    print(f"\nHead final-layer weight norms BEFORE pretraining:")
    for name, norm in norms_before.items():
        print(f"  {name}_head: {norm:.6f}  "
              f"{'(zero-init ✓)' if norm < 1e-4 else '(already trained)'}")

    # ---- Replay buffer ----
    print(f"\nSeeding replay buffer from: {args.hdf5}")
    replay = ReplayBuffer(
        capacity   = 50_000,
        n_ctx      = 4,
        action_dim = ACTION_DIM,
        gamma      = 0.99,
    )
    replay.seed_from_hdf5(args.hdf5)

    # ---- Amp dtype ----
    use_bf16  = torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    print(f"AMP dtype: {amp_dtype}")

    # ---- Build minimal args namespace for pretrain_heads ----
    head_args = argparse.Namespace(
        pretrain_heads_steps = args.steps,
        batch_size           = args.batch_size,
        lr                   = args.lr,
        weight_decay         = args.weight_decay,
    )

    # ---- Run head pretraining ----
    pretrain_heads(model, replay, head_args, device, amp_dtype)

    # ---- Head norms after training ----
    norms_after = _head_norm(model)
    print(f"Head final-layer weight norms AFTER pretraining:")
    for name, norm in norms_after.items():
        moved = norm - norms_before[name]
        print(f"  {name}_head: {norm:.6f}  (Δ={moved:+.6f})")
        if norm < 1e-4:
            print(f"    WARNING: {name}_head still near zero — "
                  "check that the HDF5 has reward labels.")

    # ---- Save output checkpoint ----
    out_path = args.output
    if out_path is None:
        stem     = Path(args.checkpoint).stem
        out_path = str(Path(args.checkpoint).parent / f"{stem}_heads.pt")

    save_dict = {k: v for k, v in ckpt.items() if k != "model"}
    save_dict["model"]            = model.state_dict()
    save_dict["heads_pretrained"] = True
    torch.save(save_dict, out_path)
    print(f"\nSaved: {out_path}")
    print("Keys:", list(save_dict.keys()))
    print("\nNext step: run online training with this checkpoint:")
    print(f"  python -m training.train_online \\")
    print(f"      --resume      {out_path} \\")
    print(f"      --hdf5        {args.hdf5} \\")
    print(f"      --score_fn    reward_value \\")
    print(f"      --backbone_lr_scale 0.1")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Pre-train reward/done/value heads")
    p.add_argument("--checkpoint",   type=str, required=True,
                   help="Path to offline checkpoint (e.g. offline_best.pt)")
    p.add_argument("--hdf5",         type=str, required=True,
                   help="Path to offline trajectory HDF5 file")
    p.add_argument("--output",       type=str, default=None,
                   help="Output path (default: <checkpoint stem>_heads.pt)")
    p.add_argument("--steps",        type=int,   default=2000)
    p.add_argument("--batch_size",   type=int,   default=128)
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
