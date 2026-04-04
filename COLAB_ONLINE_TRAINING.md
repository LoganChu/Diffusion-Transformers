# Colab Online Training Instructions

Online training closes the loop between the world model and the real ManiSkill
environment. It trains the **RewardHead**, **DoneHead**, and **ValueHead** using
live environment rewards, enabling the CEM planner to score candidate trajectories
with learned task signals instead of the proxy cube-height scorer.

**Run offline pre-training first.** This guide assumes `checkpoints/best.pt`
(or a Drive path) exists from `COLAB_TRAINING.md`.

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Offline checkpoint | `checkpoints/best.pt` — must be pre-trained |
| Trajectory HDF5 | Same file used for offline training (seeds the replay buffer) |
| ManiSkill 3 | Installed in the session (see Setup below) |
| Cosmos encoder weights | Auto-downloaded on first run (~1 GB) |

---

## 1. Setup Cell

```python
# Clone repo (or upload zip)
!git clone https://github.com/<your-username>/Diffusion-Transformers.git
%cd Diffusion-Transformers

# Install core dependencies
!pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
!pip install h5py wandb imageio numpy

# Install ManiSkill 3 (required for online env interaction)
!pip install mani-skill

# Login to WandB
import wandb
wandb.login()
```

---

## 2. Mount Drive and Verify Checkpoint

```python
from google.colab import drive
drive.mount('/content/drive')

# Verify offline checkpoint exists
import os
ckpt_path = "/content/drive/MyDrive/checkpoints/best.pt"
assert os.path.isfile(ckpt_path), f"Checkpoint not found: {ckpt_path}"
print("Checkpoint found.")

# Link trajectory HDF5
!ln -sf "/content/drive/MyDrive/trajectories_10k.h5" trajectories_10k.h5
```

---

## 3. Verify GPU

```python
!nvidia-smi
import torch
print(f"CUDA: {torch.cuda.get_device_name(0)}")
print(f"bf16: {torch.cuda.is_bf16_supported()}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

---

## 4. Phase 1 — Online Training with Proxy Scorer

Run first, using the cube-height proxy scorer (`--score_fn cube_pos`).
This trains the RewardHead, DoneHead, and ValueHead from real env rewards
while the planner is still guided by the pre-trained CubePosHead.

```bash
!python -m training.train_online \
    --hdf5        trajectories_10k.h5 \
    --resume      /content/drive/MyDrive/checkpoints/best.pt \
    --n_episodes  500 \
    --score_fn    cube_pos \
    --ckpt_dir    /content/drive/MyDrive/checkpoints_online \
    --wandb_project optiworld-online \
    --wandb_run_name "online-phase1-cube-pos"
```

**Gate before Phase 2:** Wait until `eval/success_rate > 0` appears in WandB
(typically ~50–100 episodes). This confirms the reward/value heads have learned
a meaningful signal from real env rewards.

---

## 5. Phase 2 — Upgrade to Learned Scorer

Once `eval/success_rate > 0`, switch to the trained `reward_value` scorer.
This uses `RewardHead + γ^H × ValueHead` to score CEM candidates — the full
return-bootstrapped objective.

```bash
!python -m training.train_online \
    --hdf5        trajectories_10k.h5 \
    --resume      /content/drive/MyDrive/checkpoints_online/online_best.pt \
    --n_episodes  500 \
    --score_fn    reward_value \
    --ckpt_dir    /content/drive/MyDrive/checkpoints_online \
    --wandb_project optiworld-online \
    --wandb_run_name "online-phase2-rv-scorer"
```

---

## 6. Monitor

Watch the WandB dashboard for:

| Metric | What to expect |
|---|---|
| `collect/success` | Starts 0; should reach 1 occasionally within 50–100 episodes |
| `eval/success_rate` | Key signal — phase transition happens when this first goes > 0 |
| `eval/mean_reward` | Should increase steadily; dense reward from reach + height + bonus |
| `train/loss_reward` | Should drop rapidly in first 20–50 episodes |
| `train/loss_value` | Slower to converge; large initial error as MC returns are noisy |
| `train/loss_done` | Should stabilise quickly (most steps are not done) |
| `train/loss_cfm` | Should remain close to offline val_loss; large increase = catastrophic forgetting |
| `collect/steps` | Average episode length; decreasing = faster task completion |
| `replay/size` | Grows until capacity (50K) then stabilises |

**Catastrophic forgetting check:** If `train/loss_cfm` rises sharply above the
offline baseline, the world model dynamics are being overwritten. Lower `--lr`
(try `1e-4`) or reduce `--updates_per_ep` (try 10).

---

## 7. Key Parameters

| Parameter | Default | Notes |
|---|---|---|
| `--n_episodes` | 500 | Total env episodes to collect and train on |
| `--score_fn` | `cube_pos` | Use `reward_value` after `eval/success_rate > 0` |
| `--horizon` | 6 | CEM planning horizon in world-model steps |
| `--n_candidates` | 32 | CEM candidate action sequences per iteration |
| `--n_elites` | 6 | Top-K sequences used to refit CEM distribution |
| `--n_cem_iters` | 3 | CEM refinement iterations per control step |
| `--num_ode_steps` | 4 | Euler steps per latent transition (speed vs quality) |
| `--updates_per_ep` | 20 | Gradient updates per episode on replay samples |
| `--batch_size` | 128 | Replay batch size per gradient update |
| `--lr` | 3e-4 | Lower than offline (3e-3) to avoid overwriting pretrained weights |
| `--replay_capacity` | 50000 | Max transitions stored (~360 MB CPU RAM) |
| `--min_replay_size` | 1000 | Don't start gradient updates until buffer has this many transitions |
| `--expl_noise` | 0.02 | Gaussian noise added to planned actions for exploration |
| `--eval_every` | 20 | Evaluate greedy policy every N episodes |
| `--eval_episodes` | 10 | Episodes used per evaluation |

---

## 8. Resume After Session Reset

Checkpoints save to Drive automatically. Resume is safe:

```bash
!python -m training.train_online \
    --hdf5        trajectories_10k.h5 \
    --resume      /content/drive/MyDrive/checkpoints_online/online_best.pt \
    --n_episodes  1000 \
    --score_fn    reward_value \
    --ckpt_dir    /content/drive/MyDrive/checkpoints_online \
    --wandb_run_name "online-phase2-resumed"
```

Note: The replay buffer is **not** saved to disk. On resume, it is re-seeded
from the offline HDF5 (fast, ~10 seconds). Online transitions collected in the
previous session are lost, but the model weights and optimizer state are
fully restored.

---

## 9. Download Best Checkpoint

```python
# Copy best online checkpoint back to Drive (already there if --ckpt_dir pointed to Drive)
!cp /content/drive/MyDrive/checkpoints_online/online_best.pt \
    "/content/drive/MyDrive/optiworld_online_best.pt"
print("Saved.")
```

---

## Expected Training Profile (A100 40GB)

| Metric | Expected |
|---|---|
| Episode collection time | ~30–90 sec/episode (CEM × 200 steps × 4 ODE steps) |
| Gradient updates/episode | 20 (configurable via `--updates_per_ep`) |
| First `eval/success_rate > 0` | ~50–100 episodes |
| Replay buffer full | ~500 episodes (50K transitions / ~100 steps/episode) |
| Stable `eval/success_rate` | ~200–400 episodes |
| VRAM (online, BS=128) | ~4–8 GB (model + small batch) |
| Phase 2 transition | When `eval/success_rate` first exceeds 0 in Phase 1 |

---

## Troubleshooting

**`eval/success_rate` stays at 0 for > 200 episodes:**
- Verify offline checkpoint quality: `val_loss < 5.0` in offline training
- Try `--expl_noise 0.05` (more exploration)
- Try `--n_candidates 64 --n_cem_iters 5` (more CEM compute)
- Check that `train/loss_reward` is decreasing — if not, the heads are not learning

**`train/loss_cfm` spikes after switching to Phase 2:**
- Reduce `--lr 1e-4` and resume from last checkpoint
- Reduce `--updates_per_ep 10` to slow down head training

**ManiSkill environment errors:**
- `mani_skill.envs` must be imported before `gym.make()` (already done in `train_online.py`)
- Use `render_backend="cpu"` (already set) to avoid Vulkan issues in Colab

**`KeyError` on HDF5 seed (`rewards`, `terminated` fields missing):**
- HDF5 was created with an older ingest.py that did not store reward labels
- The buffer seeds with zero rewards instead — training still works but
  WorldModelLoss reward/done/value targets start at zero until online data fills in
