# OptiWorld-FM: Diffusion Transformer for Robotic Latent Video Generation

A systems-optimized Diffusion Transformer (DiT) trained with Conditional Flow Matching (CFM) to predict future latent frames in robotic manipulation tasks. Built on ManiSkill 3.0 environments with NVIDIA's Cosmos-Tokenizer for latent encoding.

## Architecture

The model is a 12-layer DiT (~30M params) operating in the latent space of the Cosmos-Tokenizer-CI16x16:

```
RGB Observation [B, 3, 128, 128]
        ↓  Cosmos-VAE Encoder
Latent [B, 16, 8, 8]
        ↓  Patch Embed → 16 tokens × 384 dims
DiT Blocks (×12) with adaLN-Zero
  - Timestep conditioning (sinusoidal → MLP)
  - Action conditioning (8-dim → MLP)
  - SDPA attention (6 heads, 64 head_dim)
        ↓  Final Layer
Velocity Prediction [B, 16, 8, 8]
```

**Training objective:** Conditional Flow Matching — learn the velocity field `v(x_t, t)` that transports noise `x_0` to data `x_1` along straight-line paths.

## Project Structure

```
├── models/
│   ├── dit.py          # DiT backbone with adaLN-Zero, cached inference path
│   └── cache.py        # Zero-allocation KV cache (static prefix + iterative latents)
├── training/
│   ├── train.py        # Training loop: AdamW, cosine LR, bfloat16 AMP, WandB logging
│   └── loss.py         # CFM loss and simple ODE sampler
├── data/
│   ├── ingest.py       # High-throughput trajectory collection (698 LPS @ 128 envs)
│   └── dataset.py      # HDF5-backed trajectory dataset loader
├── inference/
│   └── sample.py       # Cached ODE sampling with zero-allocation Euler loop
├── pretrained_ckpts/
│   └── Cosmos-Tokenizer-CI16x16/   # NVIDIA latent encoder (16× spatial compression)
├── trajectories_10k.h5             # Training data (Git LFS)
└── COLAB_TRAINING.md               # Google Colab A100 training instructions
```

## Data Pipeline

Trajectory collection runs 128 parallel ManiSkill GPU environments, encodes RGB observations into latents via Cosmos-VAE on a separate CUDA stream, and writes episodes to HDF5 asynchronously:

```bash
conda run -n ml_env python -m data.ingest \
    --task PickCube-v1 \
    --num_envs 128 \
    --num_episodes 10000 \
    --output trajectories_10k.h5
```

**Performance:** 698 latent frames/sec (env_step 76–80%, vae_encode 18–22%, hdf5_write <0.2%).

Key optimizations:
- Double-buffered frame buffers to overlap `env.step()` with VAE encoding
- Separate CUDA streams for async encode
- Async HDF5 writer on a dedicated process

## Training

```bash
python -m training.train \
    --hdf5 trajectories_10k.h5 \
    --batch_size 256 \
    --epochs 100 \
    --lr 1e-3 \
    --grad_clip 1.0 \
    --warmup_steps 1000 \
    --val_every 1000 \
    --wandb_project optiworld-dit
```

Training uses bfloat16 mixed precision with cosine annealing LR and 1000-step linear warmup. Validation runs 8-step Heun ODE rollouts and logs GIFs to WandB every 1000 steps.

See [COLAB_TRAINING.md](COLAB_TRAINING.md) for Google Colab A100 setup instructions.

## Inference

The cached sampler pre-computes context frame KV pairs once, then runs a zero-allocation Euler ODE loop:

```python
from models.dit import DiTSmall
from inference.sample import sample_ode_cached

model = DiTSmall().cuda().half()
# x_ctx: [1, num_ctx, 16, 8, 8] context frames
# action: [1, 8] conditioning action
predicted_latent = sample_ode_cached(model, x_ctx, action, steps=8)
```

All inference is designed for batch size 1, CUDA-graph-safe (no `.item()`, no dynamic shapes, no tensor conditionals).

## MLSys Engineering Standards

| Requirement | Implementation |
|---|---|
| Attention kernel | `F.scaled_dot_product_attention` (SDPA) exclusively |
| Training precision | `torch.bfloat16` |
| Inference precision | `torch.float16`, targeting FP8 (e4m3) on Ada |
| Memory in ODE loop | Zero-allocation via `add_()`, `mul_()`, pre-allocated buffers |
| KV caching | Static prefix (context frames) + iterative latents (ODE steps) |
| CUDA graphs | Graph-safe sampling loop |
| Profiling | NVTX ranges via `torch.profiler.record_function` |

## Profiling

```bash
nsys profile -w true -t cuda,nvtx,osrt -o profiler/rep_%b python inference/bench.py
nvidia-smi --query-gpu=memory.used --format=csv
```

## Dependencies

- PyTorch 2.1+ with CUDA
- h5py
- wandb
- imageio
- mani_skill (ManiSkill 3.0)
- NVIDIA Cosmos-Tokenizer (`huggingface_hub`)
