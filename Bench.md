Config: BS=1, n_ctx=4, num_steps=8, warmup=5, repeats=50, dtype=torch.float16
MPC config: N=64, ode_steps=4
Roll config: n_roll_frames=8, n_ctx=4
Model params: 32,969,158

===========================================================================
  Single Inference Step  (Euler, BS=1, 8 evals)
Metric                                  KV Cache    Recompute
---------------------------------------------------------------------------
Total (ms)                                118.66       215.01
ms/step                                    14.83        26.88
ms/model_eval                              14.83        26.88
Speedup vs recompute                        1.81x
===========================================================================
  Single Inference Step  (Euler, BS=1, 4 evals)
Metric                                  KV Cache    Recompute
---------------------------------------------------------------------------
Total (ms)                                 66.07       107.51
ms/step                                    16.52        26.88
ms/model_eval                              16.52        26.88
Speedup vs recompute                        1.63x
===========================================================================
  Single MPC Step  (N=64, 4 evals @ BS=64)
Metric                                  KV Cache    Recompute
---------------------------------------------------------------------------
Total (ms)                                 67.78        51.05
ms/model_eval (@ BS=64)                    16.95        12.76
Speedup vs recompute                        0.75x
===========================================================================

--- Markdown (copy to results.md) ---

### Euler Solver (BS=1, 8 steps)
| Solver | Total (ms) | ms/step | ms/eval | Speedup vs recompute |
|--------|-----------|---------|---------|------------|
| Euler (KV-cached)  | 118.66 | 14.83 | 14.83 | **1.81x** |
| Euler (recompute)  | 215.01 | 26.88 | 26.88 | 1.00x |

### Euler Solver (BS=1, 4 steps)
| Solver | Total (ms) | ms/step | ms/eval | Speedup vs recompute |
|--------|-----------|---------|---------|------------|
| Euler (KV-cached)  | 66.07 | 16.52 | 16.52 | **1.63x** |
| Euler (recompute)  | 107.51 | 26.88 | 26.88 | 1.00x |

### Single MPC Step (N=64, ode=4)
| Planner | Total (ms) | ms/eval | Speedup vs recompute |
|---------|-----------|---------|------------|
| MPC (KV-cached)    | 67.78 | 16.95 | **0.75x** |
| MPC (recompute)    | 51.05 | 12.76 | 1.00x |

===========================================================================
  CUDA Graph vs Eager  (Euler step, N=64, ode=4)
Metric                                  Graph KV        Eager
---------------------------------------------------------------------------
Total (ms)                                 23.33        51.08
ms/model_eval                               5.83        12.77
Speedup vs eager                            2.19x
===========================================================================
  CUDA Graph vs Eager  (Euler BS=1, steps=8, 8 evals)
Metric                                  Graph KV     Eager KV
---------------------------------------------------------------------------
Total (ms)                                 23.70       118.66
ms/model_eval                               2.96        14.83
Speedup vs eager KV                         5.01x
===========================================================================

### CUDA Graph Speedup
| Solver | Total (ms) | ms/eval | Speedup vs eager |
|--------|-----------|---------|-----------------|
| Euler graphed KV (N=64)    | 23.33 | 5.83 | **2.19x** |
| Euler eager   (N=64)    | 51.08 | 12.77 | 1.00x |
| Euler graphed KV (BS=1)           | 23.70 | 2.96 | **5.01x** |
| Euler KV-cached eager (BS=1)   | 118.66 | 14.83 | 1.00x |

===========================================================================
  Rolling Context Slide  (8 frames, n_ctx=4)
Metric                                    Ring   Physical
---------------------------------------------------------------------------
Total (ms)                               21.11      25.94
ms/frame                                  2.64       3.24
Speedup                                   1.23x
===========================================================================

### Rolling Context Slide (8 frames, n_ctx=4)
| Slide strategy | Total (ms) | ms/frame | Speedup |
|----------------|-----------|----------|---------|
| Ring buffer    | 21.11 | 2.64 | **1.23x** |
| Physical shift | 25.94 | 3.24 | 1.00x |

===========================================================================
  Rolling Inference  (8 frames × 4 ODE steps, n_ctx=4)
Metric                                      Ring     Physical
---------------------------------------------------------------------------
Total (ms)                                434.31       436.57
ms/frame                                   54.29        54.57
Ring speedup                                1.01x
===========================================================================

### Rolling Inference (8 frames × 4 ODE steps)
| Strategy | Total (ms) | ms/frame | Speedup |
|----------|-----------|----------|---------|
| Ring buffer    | 434.31 | 54.29 | **1.01x** |
| Physical shift | 436.57 | 54.57 | 1.00x |