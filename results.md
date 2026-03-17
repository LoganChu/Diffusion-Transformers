# KV-Cache Inference Benchmark Results

## Setup

- **GPU**: NVIDIA RTX 5070 Laptop (8 GB VRAM)
- **Precision**: float16 with `torch.amp.autocast`
- **Model**: DiTSmall (32.7M params, 12 layers, 6 heads, 384 hidden)
- **Batch size**: 1
- **Context frames**: 2 (32 cached KV tokens)
- **ODE steps**: 8 (Heun's method — 15 model evaluations)
- **Warmup**: 10 runs, **Repeats**: 50 runs

## Heun Solver Latency (BS=1)

| Solver | Total (ms) | ms/step | ms/eval | Speedup |
|--------|-----------|---------|---------|---------|
| Heun (KV-cached) | 791.38 | 98.92 | 52.76 | **1.51x** |
| Heun (recompute) | 1193.81 | 149.23 | 79.59 | 1.00x |

## Analysis

The KV-cached Heun solver achieves a **1.51x** end-to-end speedup over the
recompute baseline at batch size 1.  The savings come from amortising the
context-frame prefill: with 2 context frames (32 tokens) and 8 ODE steps
(15 evaluations), the cached path runs prefill once instead of 15 times.

The per-evaluation cost drops from **79.59 ms** → **52.76 ms** (33.7% reduction),
reflecting the eliminated redundant context K/V computation at each step.

## Reproduce

```bash
python -m inference.bench --num_steps 8 --n_ctx 2 --warmup 10 --repeats 50
```
