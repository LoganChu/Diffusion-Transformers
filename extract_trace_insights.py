"""Extract detailed profiling insights from traces that demonstrate optimization mechanics."""

import json
import os
from pathlib import Path
from collections import defaultdict
from statistics import mean

TRACES_DIR = Path("traces")


def parse_trace_comprehensive(filepath: Path) -> dict:
    """Extract detailed performance metrics from a trace file."""
    try:
        with open(filepath) as f:
            data = json.load(f)

        events = data.get("traceEvents", [])

        # Extract timing breakdown
        metrics = {
            "total_duration_ms": 0,
            "kernel_ops": defaultdict(float),  # kernel name -> total duration
            "cpu_ops": defaultdict(float),     # cpu op name -> total duration
            "memory_ops": defaultdict(int),    # memory op name -> count
            "cuda_ops_count": 0,
            "cpu_ops_count": 0,
            "memory_allocs": 0,
            "memory_copies": 0,
            "peak_memory_kb": 0,
            "op_sequence": [],  # Top ops by duration
        }

        # Sum up durations by category
        for event in events:
            cat = event.get("cat", "")
            name = event.get("name", "")
            dur = event.get("dur", 0) / 1000  # Convert to ms

            if cat == "user_annotation" and "bench." in name:
                metrics["total_duration_ms"] = dur

            elif cat == "gpu_kernel":
                metrics["cuda_ops_count"] += 1
                metrics["kernel_ops"][name] += dur

            elif cat == "cpu_op":
                metrics["cpu_ops_count"] += 1
                metrics["cpu_ops"][name] += dur

                # Track memory operations
                if "empty" in name or "zeros" in name or "allocate" in name:
                    metrics["memory_allocs"] += 1
                elif "copy" in name or "to" in name or "memcpy" in name:
                    metrics["memory_copies"] += 1

        # Find top operations
        all_ops = []
        for name, dur in metrics["kernel_ops"].items():
            all_ops.append((name, dur, "kernel"))
        for name, dur in metrics["cpu_ops"].items():
            if dur > 0.01:  # Only significant CPU ops
                all_ops.append((name, dur, "cpu"))

        all_ops.sort(key=lambda x: x[1], reverse=True)
        metrics["op_sequence"] = all_ops[:10]  # Top 10 ops

        return metrics

    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return None


def categorize_benchmark(filename: str) -> tuple[str, str]:
    """Categorize and parse benchmark name."""
    base = filename.replace(".pt.trace.json", "").split(".")[0]

    if "euler_bs1_kv" in base:
        return "KV Caching (Euler BS=1)", base
    elif "euler_bs1_recompute" in base:
        return "KV Caching (Euler BS=1)", base
    elif "mpc_kv" in base:
        return "KV Caching (MPC)", base
    elif "mpc_recompute" in base:
        return "KV Caching (MPC)", base
    elif "graphed_euler_kv" in base:
        return "CUDA Graphs (Euler)", base
    elif "ungraphed_euler" in base:
        return "CUDA Graphs (Euler)", base
    elif "graphed_heun_kv" in base:
        return "CUDA Graphs (Heun)", base
    elif "slide_ring" in base:
        return "Rolling Context (Slide)", base
    elif "slide_physical" in base:
        return "Rolling Context (Slide)", base
    elif "rolling_ring" in base:
        return "Rolling Context (Inference)", base
    elif "rolling_physical" in base:
        return "Rolling Context (Inference)", base
    else:
        return "Other", base


def main():
    trace_files = sorted(TRACES_DIR.glob("*.pt.trace.json"))
    results = defaultdict(lambda: defaultdict(list))

    print(f"Parsing {len(trace_files)} traces...\n")

    for trace_file in trace_files:
        metrics = parse_trace_comprehensive(trace_file)
        if metrics:
            category, variant = categorize_benchmark(trace_file.name)
            results[category][variant].append(metrics)

    # Print insights by optimization type
    print("=" * 100)
    print("DETAILED PROFILING INSIGHTS")
    print("=" * 100)

    # 1. KV Caching insights
    print("\n" + "-" * 100)
    print("1. KV CACHING OPTIMIZATION")
    print("-" * 100)

    if "KV Caching (Euler BS=1)" in results:
        cached_runs = results["KV Caching (Euler BS=1)"]["euler_bs1_kv"]
        recompute_runs = results["KV Caching (Euler BS=1)"]["euler_bs1_recompute"]

        cached_time = mean([m["total_duration_ms"] for m in cached_runs])
        recompute_time = mean([m["total_duration_ms"] for m in recompute_runs])

        cached_kernels = mean([m["cuda_ops_count"] for m in cached_runs])
        recompute_kernels = mean([m["cuda_ops_count"] for m in recompute_runs])

        cached_cpu_ops = mean([m["cpu_ops_count"] for m in cached_runs])
        recompute_cpu_ops = mean([m["cpu_ops_count"] for m in recompute_runs])

        cached_allocs = mean([m["memory_allocs"] for m in cached_runs])
        recompute_allocs = mean([m["memory_allocs"] for m in recompute_runs])

        print(f"\nEuler (BS=1, 8 steps) with KV Caching:")
        print(f"  +- Total Time:        {cached_time:8.2f}ms (vs {recompute_time:8.2f}ms recompute)")
        print(f"  +- Speedup:           {recompute_time/cached_time:8.2f}x")
        print(f"  +- CPU Ops:           {cached_cpu_ops:.0f} ops (vs {recompute_cpu_ops:.0f})")
        if cached_allocs > 0 and recompute_allocs > 0:
            print(f"  +- Memory Allocs:     {cached_allocs:.0f} allocations (vs {recompute_allocs:.0f})")
            print(f"  |   +- Reduction:     {(1 - cached_allocs/recompute_allocs)*100:.1f}% fewer allocations")
        if cached_kernels > 0 and recompute_kernels > 0:
            print(f"  +- CUDA Kernels:      {cached_kernels:.0f} kernels (vs {recompute_kernels:.0f})")
            print(f"  |   +- Reduction:     {(1 - cached_kernels/recompute_kernels)*100:.1f}% fewer kernel launches")

        # Show top operations
        print(f"\n  Top Operations (KV Cache):")
        if cached_runs and cached_runs[0]["op_sequence"]:
            for i, (op_name, dur, op_type) in enumerate(cached_runs[0]["op_sequence"][:5], 1):
                print(f"    {i}. {op_name:<40} {dur:7.2f}ms ({op_type})")

    if "KV Caching (MPC)" in results:
        mpc_variants = results["KV Caching (MPC)"]
        mpc_cached = next(
            (runs for var, runs in mpc_variants.items() if "recompute" not in var),
            []
        )
        mpc_recompute = next(
            (runs for var, runs in mpc_variants.items() if "recompute" in var),
            []
        )

        if mpc_cached and mpc_recompute:
            cached_time = mean([m["total_duration_ms"] for m in mpc_cached])
            recompute_time = mean([m["total_duration_ms"] for m in mpc_recompute])
            cached_kernels = mean([m["cuda_ops_count"] for m in mpc_cached])
            recompute_kernels = mean([m["cuda_ops_count"] for m in mpc_recompute])
            cached_allocs = mean([m["memory_allocs"] for m in mpc_cached])
            recompute_allocs = mean([m["memory_allocs"] for m in mpc_recompute])

            print(f"\nMPC (N=64, 4 ODE steps) with KV Caching:")
            print(f"  +- Total Time:        {cached_time:8.2f}ms (vs {recompute_time:8.2f}ms recompute)")
            print(f"  +- Speedup:           {recompute_time/cached_time:8.2f}x")
            if cached_allocs > 0 and recompute_allocs > 0:
                print(f"  +- Memory Allocs:     {cached_allocs:.0f} allocations (vs {recompute_allocs:.0f})")
                print(f"  +- Reduction:         {(1 - cached_allocs/recompute_allocs)*100:.1f}% fewer allocations")
            if cached_kernels > 0 and recompute_kernels > 0:
                print(f"  +- CUDA Kernels:      {cached_kernels:.0f} kernels (vs {recompute_kernels:.0f})")
                print(f"  |   +- Reduction:     {(1 - cached_kernels/recompute_kernels)*100:.1f}% fewer kernel launches")

    # 2. CUDA Graphs insights
    print("\n" + "-" * 100)
    print("2. CUDA GRAPH OPTIMIZATION")
    print("-" * 100)

    if "CUDA Graphs (Euler)" in results:
        graph_variants = results["CUDA Graphs (Euler)"]
        graphed = graph_variants.get("graphed_euler_kv", [])
        ungraphed = graph_variants.get("ungraphed_euler", [])

        if graphed and ungraphed:
            graphed_time = mean([m["total_duration_ms"] for m in graphed])
            ungraphed_time = mean([m["total_duration_ms"] for m in ungraphed])
            graphed_kernels = mean([m["cuda_ops_count"] for m in graphed])
            ungraphed_kernels = mean([m["cuda_ops_count"] for m in ungraphed])

            print(f"\nEuler Step (N=64, 4 ODE steps) with CUDA Graphs:")
            print(f"  +- Total Time:        {graphed_time:8.2f}ms (vs {ungraphed_time:8.2f}ms eager)")
            print(f"  +- Speedup:           {ungraphed_time/graphed_time:8.2f}x")
            print(f"  +- Overhead Saved:    {(ungraphed_time - graphed_time):8.2f}ms (eliminated host-device sync)")
            if graphed_kernels > 0 and ungraphed_kernels > 0:
                print(f"  +- CUDA Kernels:      {graphed_kernels:.0f} kernels (vs {ungraphed_kernels:.0f})")
                print(f"  |   +- Reduction:     {(1 - graphed_kernels/ungraphed_kernels)*100:.1f}% fewer kernel launches")

            print(f"\n  Top Operations (Graphed):")
            if graphed and graphed[0]["op_sequence"]:
                for i, (op_name, dur, op_type) in enumerate(graphed[0]["op_sequence"][:5], 1):
                    print(f"    {i}. {op_name:<40} {dur:7.2f}ms ({op_type})")

    # 3. Rolling Context insights
    print("\n" + "-" * 100)
    print("3. ROLLING CONTEXT (RING BUFFER) OPTIMIZATION")
    print("-" * 100)

    for category_name in ["Rolling Context (Slide)", "Rolling Context (Inference)"]:
        if category_name in results:
            variants = results[category_name]
            ring = variants.get("slide_ring", []) or variants.get("rolling_ring", [])
            physical = variants.get("slide_physical", []) or variants.get("rolling_physical", [])

            if ring and physical:
                ring_time = mean([m["total_duration_ms"] for m in ring])
                phys_time = mean([m["total_duration_ms"] for m in physical])
                ring_allocs = mean([m["memory_allocs"] for m in ring])
                phys_allocs = mean([m["memory_allocs"] for m in physical])
                ring_cpu = mean([m["cpu_ops_count"] for m in ring])
                phys_cpu = mean([m["cpu_ops_count"] for m in physical])

                print(f"\n{category_name}:")
                print(f"  +- Total Time:        {ring_time:8.2f}ms (vs {phys_time:8.2f}ms physical)")
                print(f"  +- Speedup:           {phys_time/ring_time:8.2f}x")
                print(f"  +- CPU Ops:           {ring_cpu:.0f} ops (vs {phys_cpu:.0f})")
                if ring_allocs > 0 and phys_allocs > 0:
                    print(f"  +- Memory Allocs:     {ring_allocs:.0f} allocations (vs {phys_allocs:.0f})")
                    print(f"  |   +- Reduction:     {(1 - ring_allocs/phys_allocs)*100:.1f}% fewer allocations")
                print(f"  +- Key Insight:       Ring buffer avoids expensive memcpy operations")

    print("\n" + "=" * 100)


if __name__ == "__main__":
    main()
