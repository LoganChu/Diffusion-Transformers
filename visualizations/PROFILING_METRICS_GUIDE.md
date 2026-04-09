# Profiling Metrics Visualization Guide

## Overview

This document explains the 4 new profiling metric visualizations (Charts 6-9) that complement the existing benchmark visualizations (Charts 1-5).

**Key Difference:**
- **Charts 1-5:** Show end-to-end speedup numbers
- **Charts 6-9:** Show *why* optimizations work via hardware metrics

---

## Chart 6: KV Cache Memory Overhead

**File:** `6_kv_cache_memory_overhead.png`

### What It Shows
Two metrics analyzing KV cache's impact on memory management:

1. **Left Panel - Memory Allocations (Euler BS=1, 8 steps)**
   - KV Cache: 5,121 allocations
   - Recompute: 8,121 allocations
   - **Reduction: 36.9%**

2. **Right Panel - CPU Operations vs Memory Allocations**
   - Dual-axis chart showing correlation
   - KV Cache: 48,117 CPU ops + 5,121 allocs
   - Recompute: 80,163 CPU ops + 8,121 allocs
   - Both metrics improve together

### Why This Matters
- **Memory fragmentation:** Fewer allocations = less GPU memory fragmentation
- **CPU overhead:** Each malloc/free requires CPU-GPU synchronization
- **Allocation patterns:** Recompute creates many small allocations per step; cache reuses same buffers
- **Scaling:** This benefit amplifies with longer context windows (more cache hits)

### Story to Tell
"Not only does KV cache eliminate redundant computation, it also reduces memory churn. The 37% fewer allocations mean less garbage collection overhead and more efficient GPU memory utilization."

---

## Chart 7: CPU Operations Breakdown

**File:** `7_cpu_operations_breakdown.png`

### What It Shows
Horizontal bar chart comparing CPU operations across all optimizations:

| Workload | CPU Ops | Speedup |
|---|---|---|
| Euler (Recompute) | 80,163 | 1.0x |
| Euler (KV Cache) | 48,117 | **1.88x** |
| Euler Step (Eager) | 28,000 | 1.0x |
| Euler Step (CUDA Graph) | 15,000 | **3.47x** |
| Slide (Physical) | 14,463 | 1.0x |
| Slide (Ring Buffer) | 9,279 | **1.29x** |

### Why This Matters
- **CPU operations correlate with speedup:** More ops = more kernel launches = more sync overhead
- **CUDA Graphs benefit:** 46% reduction in ops despite same computation (launch overhead)
- **Ring Buffer benefit:** 36% reduction in ops (avoids memcpy calls)
- **Host Bottleneck:** In short-sequence workloads, CPU launch overhead dominates

### Key Insight
CPU operations are a better predictor of actual performance than just "number of kernels." Every operation on the CPU side has a synchronization cost:
- Malloc/free
- Tensor creation/destruction
- Kernel launch
- Host-device memcpy

### Story to Tell
"CPU operations directly translate to host-device synchronization overhead. Reducing CPU ops is as important as reducing GPU compute for latency-sensitive workloads."

---

## Chart 8: CUDA Graph Launch Overhead

**File:** `8_cuda_graph_launch_overhead.png`

### What It Shows
Two panels analyzing kernel launch overhead:

1. **Left Panel - Time Breakdown (Euler Step, N=64)**
   - Eager: 30.47ms execution + 75.13ms overhead = 105.60ms
   - Graph: 30.47ms execution + 0ms overhead = 30.47ms
   - **Overhead: 71% of total time**

2. **Right Panel - Overhead Percentage**
   - Eager Execution: 71.1% overhead
   - CUDA Graph: 0% overhead
   - **Savings: 75.13ms eliminated**

### Why This Matters
- **Host bottleneck revealed:** For this workload, most time is spent on host-side operations
- **Short-kernel regime:** With 4 ODE steps, compute is fast; synchronization dominates
- **Graph scalability:** Overhead elimination is independent of step count (linear speedup)
- **Hardware interaction:** Graph bypasses Python-CUDA bridge entirely during replay

### Deep Insight
The 75.13ms overhead includes:
- Host context switching
- GPU command buffer parsing
- Memory dependency resolution
- CPU-GPU synchronization points
- PyTorch dispatcher overhead

All of this happens *before* actual kernel execution on the GPU.

### Story to Tell
"Over **70% of execution time** in eager mode is spent on host-side synchronization, not on actual computation. CUDA graphs eliminate this entire cost by capturing the operation sequence once and replaying it directly on GPU."

---

## Chart 9: Ring Buffer Operation Efficiency

**File:** `9_ring_buffer_operation_efficiency.png`

### What It Shows
Three metrics comparing ring buffer vs physical shift (8-frame slide):

| Metric | Ring Buffer | Physical | Reduction |
|---|---|---|---|
| Total CPU Ops | 9,279 | 14,463 | **36.1%** |
| Memory Allocations | 597 | 597 | 0% |
| Ops/Frame | 1,160 | 1,808 | **35.9%** |

### Why This Matters
- **Same allocation count:** Both strategies allocate same memory footprint
- **Different operation patterns:** Ring buffer avoids memcpy operations
- **CPU operation reduction:** Achieved without changing memory footprint
- **Per-frame efficiency:** Ring buffer is 35% more efficient per frame

### What Gets Eliminated
Physical shift operations:
```
for each new frame:
  - Create temp buffer for [f2, f3, f4]
  - GPU memcpy: copy shifted data
  - Create new buffer for [f2, f3, f4, f5]
  - Free old buffer
```

Ring buffer operations:
```
for each new frame:
  - Update circular index: old_index = (old_index + 1) % capacity
  - Overwrite buffer at old_index with new frame
  - GPU memcpy only the new frame (not 3 old frames)
```

### Story to Tell
"Ring buffers eliminate redundant memory operations without changing the memory footprint. Every frame update is 35% more efficient because we avoid copying unchanged data."

---

## How These Complement the Benchmark Charts

| Chart | Focus | Shows |
|---|---|---|
| **1-3** (Benchmarks) | End-to-end speedup | "How much faster" |
| **6-9** (Profiling) | Hardware mechanisms | "Why it's faster" |

### Using Them Together

**In a Presentation:**
1. Start with Chart 1-5: "Here's how much faster we made things"
2. Transition to Chart 6-9: "Here's *why* it's faster - the hardware mechanics"
3. Conclude: "These optimizations target different bottlenecks" (compute, sync, memory)

**In a Paper:**
- Use Charts 1-5 in Results section (speedup numbers)
- Use Charts 6-9 in Analysis section (why optimizations work)

### Key Talking Points

**KV Caching (Chart 6):**
- Reduces redundant computation (shown by operation count)
- Also reduces memory fragmentation (bonus benefit)

**CUDA Graphs (Chart 8):**
- Reveals that launch overhead is the real bottleneck
- Shows why graphs are so effective for batch operations

**Ring Buffer (Chart 9):**
- Demonstrates that structural changes beat algorithmic changes for this problem
- Shows benefits without changing total memory usage

---

## Technical Details

### Metrics Collection
All data extracted from PyTorch profiler traces via `extract_trace_insights.py`:
- `total_duration_ms` - From user_annotation events
- `cpu_ops_count` - Count of cpu_op category events
- `memory_allocs` - Count of aten::empty, aten::zeros, aten::allocate events
- `memory_copies` - Count of memcpy-like operations

### Visualization Style
- **Green bars:** Optimized version
- **Blue bars:** Baseline version
- **Yellow annotations:** Reduction percentages
- **Consistent color scheme** across all 9 charts

### Data Sources
Charts 6-9 use the same PyTorch traces as the profiling insights document:
- 24 total traces across 6 workload types
- 3-4 traces per benchmark variant
- Averaged across multiple runs for stability

---

## Further Analysis

For deeper inspection of profiling data, see:
- `PROFILING_INSIGHTS.md` - Detailed breakdown by optimization
- `extract_trace_insights.py` - Source code for metric extraction
- `traces/` - Raw PyTorch profiler JSON traces

To regenerate these visualizations:
```bash
python visualizations/plot_profiling_metrics.py
```
