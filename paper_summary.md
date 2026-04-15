# OptiWorld-FM: ML Systems Paper Summary

**Codebase:** Diffusion-Transformers
**Date:** 2026-04-13

---

## 1. System Overview

**Problem:** Real-time robot manipulation planning requires predicting future visual states (video frames) from current observations and proposed actions — fast enough to run as a model-predictive controller (MPC) at inference time.

**Core Idea:** A Diffusion Transformer (DiT) world model trained on latent video frames, combined with prefix KV-caching and CUDA graph capture to make ODE-based generation fast enough for CEM-MPC planning. The system predicts next-frame VAE latents conditioned on historical context frames and robot actions, and uses those predictions to score action sequences.

**Key novelty (systems angle):** Adapting static-prefix KV caching (a technique from LLM inference) to the diffusion ODE sampling loop, where the "prefix" is historical context frames and the "query" is a noisy latent that changes at every ODE step.

---

## 2. Architecture

### Main Components

| Component | File | Role |
|-----------|------|------|
| `DiTSmall` | `models/dit.py:351` | 12-layer adaLN-Zero DiT, ~33M params (DiT-Small) |
| `KVCache` | `models/cache.py:7` | Pre-allocated, zero-copy prefix KV cache |
| `RingKVCache` | `models/cache.py:97` | O(1) context eviction via circular head pointer |
| `GraphedEulerStep` | `inference/graph_solver.py:60` | CUDA-graph-captured Euler ODE for CEM inner loop |
| `GraphedHeunSolver` | `inference/graph_solver.py:197` | CUDA-graph-captured Euler for BS=1 high-quality inference |
| `cem_plan()` | `inference/planner.py:199` | Cross-Entropy Method MPC planner |
| `CFMLoss` | `training/loss.py:7` | Conditional Flow Matching training loss |
| `WorldModelLoss` | `training/loss.py:42` | CFM + reward/done/value heads for online RL |

### Data/Control Flow

**Inference (reactive control):**
```
[B, n_ctx, 16, 8, 8] ctx_latents
    → prefill_cache() [ONCE per frame, outside ODE loop]
        → DiTBlock.forward_prefill() × 12 layers
        → KVCache.prefill(): writes context K/V into static prefix slots
    → ODE loop (8 steps): x_noise → x_predicted
        → t_buf.fill_(i*dt)           [in-place, graph-safe]
        → model(x, t, a, cache=cache) → velocity v
            → DiTBlock.forward(): denoise Q attends over [cache.ctx_K | denoise_K]
        → x.add_(v, alpha=dt)         [in-place, graph-safe]
```

**Planning (CEM-MPC):**
```
ctx_latents [1, n_ctx, C, H, W]
    → CEM outer loop (3 iters):
        → Sample N=64 action sequences [N, H=6, 4]
        → Horizon rollout:
            → _euler_rollout_step(): NO cache, recompute ctx each step
                → model(x, t, a, ctx_latents=ctx): full self-attention
            → score_fn(z_next) → scalar
            → ctx = cat([ctx[:, 1:], z_next.unsqueeze(1)]) [rolling window]
        → Elite top-8, refit Gaussian
    → Return mean[0]: first action
```

**Training/Inference Parity:** Context as prepended tokens (training) and context as KV prefix (inference) are numerically equivalent — verified in `tests/test_parity.py`.

### Novel Abstractions

- **Split-cache layout:** Buffer is logically `[0:n_ctx]` static prefix + `[n_ctx:n_total]` denoise region. Prefill writes once; ODE loop overwrites denoise region each step via `cache.update()`.
- **`DiTBlock` dual-mode forward:** `forward()` for inference (cross-attention into cache), `forward_prefill()` for prefix construction — same module, two execution paths.
- **Ring buffer for rolling context:** `RingKVCache.slide_ring()` writes only the new frame at `head * n_frame` slot, advances pointer mod `n_frames`. O(n_frame) vs O(n_ctx) physical shift. Permutation-invariance of full (non-causal) attention makes non-chronological physical layout correct.

---

## 3. Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **adaLN-Zero conditioning** | Action + timestep injected via adaLN shift/scale/gate on every block — avoids cross-attention for conditioning, keeps compute uniform across ODE steps |
| **Cross-attention for context (inference)** | Denoise Q attends over context K/V from cache — clean separation between static prefix and dynamic query, enables KV reuse |
| **BS=1 shared cache for N=64 MPC batch** | Context K/V stored at batch=1, SDPA broadcasts to N candidates. Avoids N× memory for cache but limits throughput gains for large N |
| **CUDA Graph unrolled ODE** | Captures entire N-step ODE as one graph — eliminates Python dispatch overhead per step (71% of eager time is overhead, not compute) |
| **Zero-allocation ODE inner loop** | Pre-allocated `x`, `t_buf`, `x_euler` buffers; all ops are `add_`, `fill_`, `.copy_()` — required for CUDA graph shape stability |
| **Ring vs physical-shift KVCache** | Ring is O(n_frame) write per context update; physical shift is O(n_ctx) memcpy. Ring preferred for streaming; physical for simplicity |
| **Prefill runs outside CUDA graph** | Context changes every planning horizon step — prefill cannot be graphed with fixed context inputs. ODE loop is graphed; prefill is eager |
| **No KV cache for CEM rollout** | At N=64, compute parallelism dominates; sharing a BS=1 cache adds 33% per-eval overhead vs recompute (16.95ms vs 12.76ms) |

---

## 4. Performance-Critical Paths

### Time Distribution (from profiling traces)

**KV-cached Euler BS=1, 8 steps:**

| Op | CUDA Time | Approx % |
|----|-----------|----------|
| `aten::linear` (QKV projections + MLP) | 362.27ms | ~55% |
| `aten::layer_norm` | 160.73ms | ~24% |
| `aten::addmm` | 115.42ms | ~17% |
| `aten::scaled_dot_product_attention` | 74.74ms | ~11% |
| `aten::add` (residual) | 60.82ms | ~9% |

**Eager MPC (N=64, 4 steps) — overhead analysis:**

| Source | Time | % of Eager |
|--------|------|-----------|
| Host-device sync / kernel launch overhead | 75.13ms | 71% |
| Actual GPU compute | ~30.47ms | 29% |
| **Eager total** | **105.60ms** | 100% |

### Key Bottlenecks

1. **`aten::linear` dominates compute** — QKV projections and MLP across 12 layers × N steps
2. **Kernel launch overhead dominates short-ODE workloads** — 71% of eager time for 4-step ODE is CPU-side dispatch, not GPU compute
3. **Context prefill is the main redundancy eliminated by caching** — per-eval cost drops from 26.88ms → 14.83ms (44.8% reduction)
4. **KV cache becomes counter-productive at N=64** — 0.75x vs recompute baseline; crossover point estimated ~256 evals

---

## 5. Optimization Techniques

### Caching
- **Static-prefix KV cache:** Context frames encoded once, K/V stored in pre-allocated buffer `[depth, 1, n_heads, n_ctx, head_dim]`. Per-ODE-step: only denoise tokens compute new K/V.
- **Ring buffer eviction:** O(1) context slot update via circular pointer — eliminates O(n_ctx) memcpy for streaming scenarios.

### Parallelism
- **CUDA Graph capture:** Entire ODE unroll (N steps × model forward) captured as single graph — collapses thousands of kernel launches into one `graph.replay()` call.
- **BS=1 broadcast for CEM:** Single context cache shared across N=64 candidates via SDPA implicit broadcast.

### Memory
- **Zero-allocation ODE loop:** All tensors pre-allocated outside loop; `add_`, `fill_()`, `copy_()` used exclusively inside.
- **Float16 inference / BFloat16 training:** Reduces memory bandwidth by 2× vs float32; explicit float32 escape for sinusoidal embeddings only.
- **Replay buffer float16 latents:** CPU-resident buffer stores latents in float16 for memory efficiency.

### Kernel Selection
- **SDPA exclusively:** `torch.nn.functional.scaled_dot_product_attention` — dispatches to FlashAttention-2 or memory-efficient attention automatically.
- **`torch.compile` (max-autotune):** ~20% speedup reported on A100.

### Solver
- **Heun corrector:** 2nd-order accuracy with 2× model evals per step; pre-allocated `x_euler`, in-place `v1.add_(v2)` for corrector accumulation.
- **Euler for CEM:** 1st-order for planning (N=64, 4 steps) — accuracy less critical, throughput more important.

### Profiling Infrastructure
- NVTX `range_push/pop` + `torch.profiler.record_function` on every module and cache method — dual Nsight Systems / TensorBoard visibility.

---

## 6. Experimental Setup

### Benchmarks (from `Bench.md`, `results.md`)

| Workload | Variants Compared |
|----------|-------------------|
| BS=1 Euler, 8 ODE steps | KV cache vs recompute |
| MPC N=64, 4 ODE steps | KV cache vs recompute |
| CUDA Graph vs Eager | Both workloads |
| Rolling context, 8 frames | Ring vs physical-shift slide |
| Heun solver BS=1, 8 steps | KV cache vs recompute |

### Reported Metrics

- Wall-clock total (ms) for full workload
- ms/step (per ODE step)
- ms/eval (per model forward pass)
- Speedup vs baseline for each comparison
- CPU op counts and memory allocation counts (from profiler traces)

### Concrete Numbers

| Comparison | Result |
|------------|--------|
| KV cache vs recompute (BS=1, 8 steps) | **1.81x** speedup |
| CUDA graph vs eager (BS=1, 8 steps) | **5.01x** speedup |
| CUDA graph vs eager (N=64, 4 steps) | **2.19x** speedup |
| Heun KV cache vs recompute (BS=1) | **1.51x** speedup |
| Ring vs physical slide (isolated) | **1.23x** speedup |
| Ring vs physical slide (full rollout) | **1.01x** speedup |
| KV cache (N=64 MPC) vs recompute | **0.75x** (regression) |

### Missing Baselines

- No Flash Attention vs non-Flash baseline
- No `torch.compile` vs non-compile numbers in `Bench.md`
- No FP8 results (listed as future work in `CLAUDE.md`)
- No wall-clock comparison vs non-DiT baselines (e.g., CNN world model)
- No CEM planning success rate vs real-time latency constraint
- No memory (VRAM) usage reported — only time

---

## 7. Gaps / Weaknesses

### Missing Experiments

- **Memory footprint not benchmarked** — only latency is reported; VRAM for KV cache vs recompute is unquantified
- **No scaling curves** — single model size (DiT-Small); no ablation on depth/width vs cache efficiency tradeoff
- **No KV cache crossover curve** — batch size N at which caching inverts is inferred (~256) but not measured
- **`torch.compile` results absent from `Bench.md`** — referenced as ~20% on A100 in training code but no bench entry
- **FP8 not implemented** — named as a target in `CLAUDE.md`; would be a major result
- **No task success rate** — CEM planner quality vs planning latency tradeoff not studied

### Questionable Assumptions

- **BS=1 shared cache for N=64 MPC:** Assumes SDPA broadcast is zero-cost, but introduces BS mismatch between cache (BS=1) and compute (BS=64) — confirmed to hurt throughput at N=64
- **Prefill outside CUDA graph:** Prefill costs ~9ms per frame eagerly; capturing a per-context prefill graph would reduce this
- **Ring buffer benefit disappears at scale:** 1.23x isolated → 1.01x in full rollout — optimization is real but contribution is negligible in production workloads
- **CEM rollout has no caching:** The heaviest path (N=64 × H=6 × 4 ODE = 1,536 evals) has zero caching benefit — needs explanation or mitigation

### Unclear Design Choices

- **`GraphedHeunSolver` implements Euler, not Heun** — `graph_solver.py:256` captures an Euler loop despite "Heun" naming; should be corrected before describing solvers in a paper
- **Context K/V output discarded during prefill** — `dit.py:458` discards `x_ctx`; only K/V side-effects matter. A K/V-only path would be more efficient
- **`n_ctx=4` hardcoded** — `MAX_CTX_FRAMES=4` at `dit.py:28`; no experiment studies sensitivity to context length
- **Action conditioning via addition** — action embedding added to timestep embedding before adaLN; conflates action and time signals; no ablation vs separate conditioning

---

## Critical Files Reference

| File | Role |
|------|------|
| `models/dit.py` | Core architecture — DiTSmall, DiTBlock, dual forward modes |
| `models/cache.py` | KVCache and RingKVCache — pre-allocated buffers, slide methods |
| `inference/graph_solver.py` | CUDA graph capture — GraphedEulerStep, GraphedHeunSolver |
| `inference/solver.py` | Heun solver with zero-allocation inner loop |
| `inference/planner.py` | CEM-MPC planner with pluggable score functions |
| `inference/bench.py` | Benchmark harness — all timing measurements |
| `training/loss.py` | CFMLoss + WorldModelLoss |
| `Bench.md` | Consolidated benchmark results table |
| `PROFILING_INSIGHTS.md` | Bottleneck analysis from traces |
| `traces/` | 24 PyTorch profiler JSON traces |
