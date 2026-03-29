# Colab A100 Training Instructions

## 1. Setup Cell

```python
# Clone repo (or upload zip)
!git clone https://github.com/<your-username>/Diffusion-Transformers.git
%cd Diffusion-Transformers

# Install dependencies
!pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
!pip install h5py wandb imageio numpy

# Login to WandB
import wandb
wandb.login()
```

## 2. Upload Data

`trajectories_7k.h5` is ~4 GB. Google Drive mount is recommended over direct upload.

```python
# Option A: Google Drive (recommended — persists across sessions)
from google.colab import drive
drive.mount('/content/drive')
!ln -s "/content/drive/MyDrive/trajectories_7k.h5" trajectories_7k.h5

# Option B: Direct upload (lost when runtime resets)
from google.colab import files
uploaded = files.upload()  # select trajectories_7k.h5
```

## 3. Verify GPU

```python
!nvidia-smi
import torch
print(f"CUDA: {torch.cuda.get_device_name(0)}")
print(f"bf16: {torch.cuda.is_bf16_supported()}")  # Should be True on A100
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

## 4. Training Command

```bash
# Full training run
!python -m training.train \
    --hdf5 trajectories_7k.h5 \
    --ctx_frames 4 \
    --batch_size 256 \
    --epochs 100 \
    --lr 1e-3 \
    --weight_decay 0.05 \
    --grad_clip 1.0 \
    --warmup_steps 1000 \
    --val_every 1000 \
    --log_every 50 \
    --num_workers 4 \
    --wandb_project optiworld-dit \
    --wandb_run_name "a100-cfm-7k" \
    --compile
```

`--compile` enables `torch.compile(mode="max-autotune")` for ~20% speedup. Adds ~2–5 min overhead on the first step. Drop the flag if you hit compile errors.

### Tuning Tips

- **Loss unstable / spikes:** Gradient clipping at 1.0 and cosine schedule are on by default.
- **Loss plateaus early:** Try `--lr 5e-4` or increase `--epochs 200`.
- **Want faster training:** Increase `--batch_size 512` (A100 40GB has headroom — model is only 32.7M params).
- **OOM:** Unlikely at BS=256. If it happens drop to `--batch_size 128`.

## 5. Monitor

Watch the WandB dashboard for:
- `train/loss` — total loss (CFM + 0.1 × aux), should decrease steadily from ~1.0
- `train/loss_cfm` — flow matching loss; primary convergence signal
- `train/loss_aux` — cube position prediction loss; drops quickly (simple regression)
- `train/grad_norm` — should stay < 1.0 after warmup
- `val/loss` — watch for plateau; gap vs train loss indicates overfitting
- `val/rollout_gif` — visual quality check every 1000 steps; meaningful structure should appear by step 5000+
- `gpu/mem_peak_mb` — expected ~4–6 GB at BS=256

## 6. Resume from Checkpoint

```bash
!python -m training.train \
    --hdf5 trajectories_7k.h5 \
    --ctx_frames 4 \
    --resume checkpoints/best.pt \
    --epochs 200 \
    --wandb_run_name "a100-cfm-7k-resumed"
```

## 7. Download Results

```python
from google.colab import files
files.download('checkpoints/best.pt')
# Or copy to Drive (recommended — survives runtime reset)
!cp checkpoints/best.pt "/content/drive/MyDrive/optiworld_best.pt"
```

## Expected Training Profile (A100 40GB)

| Metric | Expected |
|--------|----------|
| Dataset samples | ~1.88M (7,237 episodes × ~264 steps, skipping first 4 per episode for ctx) |
| Steps per epoch (BS=256) | ~7,350 |
| Throughput | ~800–1200 steps/min at BS=256 |
| Memory | ~4–6 GB (32.7M param model, bfloat16) |
| AMP dtype | bfloat16 (native on A100, no GradScaler needed) |
| Convergence | Loss plateau ~50–80 epochs (~370k–590k steps) |
| Val GIFs | Meaningful structure by step 5000+ |
| Total wall time (100 epochs) | ~10–15 hours |
