# Colab A100 Training Instructions

## 1. Setup Cell

```python
# Clone repo (or upload zip)
!git clone https://github.com/<your-username>/Diffusion-Transformers.git
%cd Diffusion-Transformers

# Install dependencies
!pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
!pip install h5py wandb imageio

# Login to WandB
import wandb
wandb.login()
```

## 2. Upload Data

Upload `trajectories_10k_v2.h5` (the updated dataset with `ee_pos`, `cube_pos`, `phase`) to the Colab runtime or mount Google Drive:

```python
# Option A: Google Drive
from google.colab import drive
drive.mount('/content/drive')
!ln -s /content/drive/MyDrive/trajectories_10k_v2.h5 trajectories_10k_v2.h5

# Option B: Direct upload
from google.colab import files
uploaded = files.upload()  # select trajectories_10k_v2.h5
```

## 3. Verify GPU

```python
!nvidia-smi
import torch
print(f"CUDA: {torch.cuda.get_device_name(0)}")
print(f"bf16: {torch.cuda.is_bf16_supported()}")  # Should be True on A100
```

## 4. Training Command

```bash
# Full training run
!python -m training.train \
    --hdf5 trajectories_10k_v2.h5 \
    --batch_size 256 \
    --epochs 100 \
    --lr 1e-3 \
    --grad_clip 1.0 \
    --warmup_steps 1000 \
    --val_every 1000 \
    --log_every 50 \
    --num_workers 2 \
    --wandb_project optiworld-dit \
    --wandb_run_name "a100-cfm-v2-state"
```

### Tuning Tips

- **Loss unstable / spikes:** Already handled — gradient clipping at 1.0 and cosine schedule are on by default.
- **Loss plateaus too early:** Try `--lr 5e-4` or increase `--epochs 200`.
- **OOM on A100 (40GB):** Unlikely at BS=256 for this model (~30M params). If it happens, drop to `--batch_size 128`.
- **Faster convergence:** The cosine schedule with 1000-step warmup should give smooth convergence.

## 5. Monitor

Watch the WandB dashboard for:
- `train/loss` — total loss (CFM + 0.1 × aux), should decrease steadily
- `train/loss_cfm` — flow matching loss; primary convergence signal
- `train/loss_aux` — cube position prediction loss; should drop quickly (simple regression)
- `train/grad_norm` — should stay < 1.0 after warmup
- `val/loss` — track for plateau detection
- `val/rollout_gif` — visual quality check every 1000 steps
- `gpu/mem_peak_mb` — memory utilization (~4-6 GB, slightly higher than before due to CubePosHead)

## 6. Resume from Checkpoint

```bash
!python -m training.train \
    --hdf5 trajectories_10k_v2.h5 \
    --resume checkpoints/best.pt \
    --epochs 200 \
    --wandb_run_name "a100-cfm-v2-state-resumed"
```

## 7. Download Results

```python
from google.colab import files
files.download('checkpoints/best.pt')
# Or copy to Drive
!cp checkpoints/best.pt /content/drive/MyDrive/
```

## Expected Training Profile (A100)

| Metric | Expected |
|--------|----------|
| Throughput | ~3000 steps/sec at BS=256 |
| Memory | ~4-6 GB (small model) |
| AMP dtype | bfloat16 (native on A100) |
| Convergence | Loss plateau ~50-80 epochs |
| Val GIFs | Meaningful structure by step 5000+ |
