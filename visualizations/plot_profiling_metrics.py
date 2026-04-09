"""
Publication-quality profiling metric visualizations for optimization analysis.
Complements benchmark speedups with deeper profiling insights.

Run: python visualizations/plot_profiling_metrics.py

Generates:
6. kv_cache_memory_overhead.png
7. cpu_operations_breakdown.png
8. cuda_graph_launch_overhead.png
9. ring_buffer_operation_efficiency.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle
import os

os.makedirs("visualizations/output", exist_ok=True)

# Color scheme
COLOR_SUCCESS = "#2ecc71"     # green
COLOR_FAIL = "#e74c3c"        # red
COLOR_NEUTRAL = "#3498db"     # blue
COLOR_ACCENT = "#f39c12"      # orange
COLOR_SECONDARY = "#9b59b6"   # purple
COLOR_LIGHT = "#ecf0f1"       # light gray


# ============================================================================
# Chart 6: KV Cache Memory Allocation Overhead
# ============================================================================

def plot_kv_cache_memory_overhead():
    """Show memory allocation and copy reduction from KV caching."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Euler BS=1 - Memory Allocations
    scenarios_euler = ["KV Cache", "No KV Cache"]
    allocs_euler = [5121, 8121]
    colors_euler = [COLOR_SUCCESS, COLOR_NEUTRAL]

    bars1 = ax1.bar(scenarios_euler, allocs_euler, color=colors_euler, edgecolor='black',
                    linewidth=2, alpha=0.8, width=0.5)

    # Add labels
    ax1.text(0, 5121 + 200, "5,121", ha='center', fontsize=12, fontweight='bold')
    ax1.text(1, 8121 + 200, "8,121", ha='center', fontsize=12, fontweight='bold')

    # Highlight reduction
    reduction_pct = ((8121 - 5121) / 8121) * 100
    ax1.text(0.5, 6500, f"{reduction_pct:.1f}%\nfewer allocs", ha='center', fontsize=13,
            fontweight='bold', bbox=dict(boxstyle='round', facecolor=COLOR_SUCCESS, alpha=0.3))

    ax1.set_ylabel("Memory Allocation Count", fontsize=12, fontweight='bold')
    ax1.set_title("Memory Allocations", fontsize=13, fontweight='bold')
    ax1.set_ylim(0, 9500)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    # Right: Memory Copies only
    workloads = ["KV Cache", "No KV Cache"]
    mem_copies = [52641, 102572]

    bars_mem = ax2.bar(workloads, mem_copies, color=[COLOR_SUCCESS, COLOR_NEUTRAL],
                       edgecolor='black', linewidth=2, alpha=0.8, width=0.5)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars_mem, mem_copies)):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 2000, f"{val:,.0f}",
                ha='center', fontsize=12, fontweight='bold')

    # Highlight reduction
    mem_reduction_pct = ((102572 - 52641) / 102572) * 100
    ax2.text(0.5, 75000, f"{mem_reduction_pct:.1f}%\nfewer copies", ha='center', fontsize=13,
            fontweight='bold', bbox=dict(boxstyle='round', facecolor=COLOR_SUCCESS, alpha=0.3))

    ax2.set_ylabel("Memory Copy Operations", fontsize=12, fontweight='bold')
    ax2.set_title("Memory Copy Operations", fontsize=13, fontweight='bold')
    ax2.set_ylim(0, 120000)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    fig.suptitle("KV Cache: Memory Efficiency Gains",
                 fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout()
    plt.savefig("visualizations/output/6_kv_cache_memory_overhead.png", dpi=300, bbox_inches='tight')
    plt.savefig("visualizations/output/6_kv_cache_memory_overhead.pdf", bbox_inches='tight')
    print("[OK] Saved: 6_kv_cache_memory_overhead.png")
    print("[OK] Saved: 6_kv_cache_memory_overhead.pdf")
    plt.close()


# ============================================================================
# Chart 7: CPU Operations Breakdown by Optimization
# ============================================================================

def plot_cpu_operations_breakdown():
    """Show how CPU operations vary across optimizations."""
    fig, ax = plt.subplots(figsize=(12, 6))

    optimizations = [
        "Euler\n(Recompute)",
        "Euler\n(KV Cache)",
        "Euler Step\n(Eager)",
        "Euler Step\n(CUDA Graph)",
        "Slide\n(Physical)",
        "Slide\n(Ring Buffer)",
    ]

    cpu_ops = [80163, 48117, 28000, 15000, 14463, 9279]
    speedups = [1.0, 1.88, 1.0, 3.47, 1.0, 1.29]

    # Color by optimization type
    colors_list = [
        COLOR_NEUTRAL,      # KV recompute
        COLOR_SUCCESS,      # KV cache
        COLOR_NEUTRAL,      # eager
        COLOR_SUCCESS,      # graph
        COLOR_NEUTRAL,      # physical
        COLOR_SUCCESS,      # ring
    ]

    bars = ax.barh(optimizations, cpu_ops, color=colors_list, edgecolor='black',
                   linewidth=2, alpha=0.8)

    # Add value labels and speedups
    for i, (bar, ops, speedup) in enumerate(zip(bars, cpu_ops, speedups)):
        ax.text(ops + 1500, bar.get_y() + bar.get_height()/2,
               f"{ops:,.0f} ops", ha='left', va='center', fontsize=10, fontweight='bold')

        # Add speedup annotation
        speedup_color = COLOR_SUCCESS if speedup > 1.2 else COLOR_NEUTRAL
        ax.text(ops - 5000, bar.get_y() + bar.get_height()/2,
               f"{speedup:.2f}x", ha='right', va='center', fontsize=11, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor=speedup_color, alpha=0.2))

    ax.set_xlabel("CPU Operations Count", fontsize=12, fontweight='bold')
    ax.set_title("CPU Operations Overhead by Optimization\n(Lower = Better, Less Launch Overhead)",
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xlim(0, 100000)
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    # Add legend in top right to avoid blocking numbers
    green_patch = mpatches.Patch(color=COLOR_SUCCESS, label='Optimized', alpha=0.8, edgecolor='black', linewidth=1.5)
    blue_patch = mpatches.Patch(color=COLOR_NEUTRAL, label='Baseline', alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.legend(handles=[green_patch, blue_patch], loc='upper right', fontsize=11, framealpha=0.95)

    plt.tight_layout()
    plt.savefig("visualizations/output/7_cpu_operations_breakdown.png", dpi=300, bbox_inches='tight')
    plt.savefig("visualizations/output/7_cpu_operations_breakdown.pdf", bbox_inches='tight')
    print("[OK] Saved: 7_cpu_operations_breakdown.png")
    print("[OK] Saved: 7_cpu_operations_breakdown.pdf")
    plt.close()


# ============================================================================
# Chart 8: CUDA Graph Launch Overhead Elimination
# ============================================================================

def plot_cuda_graph_launch_overhead():
    """Show breakdown of launch overhead eliminated by CUDA graphs."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Stacked bar showing execution vs overhead breakdown
    scenarios = ['Eager\nExecution', 'CUDA\nGraph']
    exec_time = [30.47, 30.47]
    overhead_time = [75.13, 0]

    x = np.arange(len(scenarios))
    width = 0.5

    # Stacked bars
    bars1 = ax1.bar(x, exec_time, width, label='Kernel Execution', color=COLOR_ACCENT,
                   edgecolor='black', linewidth=2, alpha=0.8)
    bars2 = ax1.bar(x, overhead_time, width, bottom=exec_time, label='Host Overhead',
                   color=COLOR_FAIL, edgecolor='black', linewidth=2, alpha=0.8)

    # Labels on segments
    for i, (ex, oh) in enumerate(zip(exec_time, overhead_time)):
        # Execution time label
        ax1.text(i, ex/2, f'{ex:.1f}ms\nexec', ha='center', va='center',
                fontsize=11, fontweight='bold', color='white')
        # Overhead label (only if > 0)
        if oh > 0:
            ax1.text(i, ex + oh/2, f'{oh:.1f}ms\noverhead', ha='center', va='center',
                    fontsize=11, fontweight='bold', color='white')

    # Total time labels on top
    for i, (ex, oh) in enumerate(zip(exec_time, overhead_time)):
        total = ex + oh
        ax1.text(i, total + 3, f'{total:.1f}ms total', ha='center', va='bottom',
                fontsize=11, fontweight='bold')

    ax1.set_ylabel('Time (ms)', fontsize=12, fontweight='bold')
    ax1.set_title('Single Trajectory Time Breakdown', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenarios)
    ax1.set_ylim(0, 125)
    ax1.legend(fontsize=11, loc='upper right')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    # Right: Overhead percentage
    scenarios = ["Eager\nExecution", "CUDA\nGraph"]
    overhead_pcts = [71.1, 0]  # 75.13 / 105.6 = 71.1%
    colors = [COLOR_FAIL, COLOR_SUCCESS]

    bars = ax2.bar(scenarios, overhead_pcts, color=colors, edgecolor='black',
                   linewidth=2, alpha=0.8, width=0.5)

    ax2.text(0, 71.1 + 3, "71.1%\noverhead", ha='center', fontsize=12, fontweight='bold')
    ax2.text(1, 0 + 3, "0%\noverhead", ha='center', fontsize=12, fontweight='bold')

    # Highlight savings
    ax2.annotate('', xy=(1, 35), xytext=(0, 35),
                arrowprops=dict(arrowstyle='<->', color='black', lw=2))
    ax2.text(0.5, 40, "71.1% overhead\neliminated", ha='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor=COLOR_SUCCESS, alpha=0.3))

    ax2.set_ylabel("Launch Overhead (%)", fontsize=12, fontweight='bold')
    ax2.set_title("Host-Device Synchronization Overhead\n(% of Total Execution)",
                 fontsize=13, fontweight='bold')
    ax2.set_ylim(0, 100)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    fig.suptitle("CUDA Graphs: Launch Overhead Elimination",
                fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout()
    plt.savefig("visualizations/output/8_cuda_graph_launch_overhead.png", dpi=300, bbox_inches='tight')
    plt.savefig("visualizations/output/8_cuda_graph_launch_overhead.pdf", bbox_inches='tight')
    print("[OK] Saved: 8_cuda_graph_launch_overhead.png")
    print("[OK] Saved: 8_cuda_graph_launch_overhead.pdf")
    plt.close()


# ============================================================================
# Chart 9: Ring Buffer Operation Efficiency
# ============================================================================

def plot_ring_buffer_operation_efficiency():
    """Show CPU operation reduction in ring buffer optimization."""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_axes([0.1, 0.12, 0.8, 0.75])  # Leave space at bottom for insight text

    # Data: operation counts for ring vs physical
    operations = [
        "Total CPU\nOperations",
        "Memory\nAllocations",
        "Effective\nOps/Frame",
    ]

    ring_ops = [9279, 597, 9279/8]        # 8 frames
    physical_ops = [14463, 597, 14463/8]   # Same allocation count but more CPU ops

    x = np.arange(len(operations))
    width = 0.35

    bars1 = ax.bar(x - width/2, ring_ops, width, label='Ring Buffer', color=COLOR_SUCCESS,
                  edgecolor='black', linewidth=1.5, alpha=0.8)
    bars2 = ax.bar(x + width/2, physical_ops, width, label='Sliding Window', color=COLOR_NEUTRAL,
                  edgecolor='black', linewidth=1.5, alpha=0.8)

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars1, ring_ops)):
        height = bar.get_height()
        offset = 400 if val > 5000 else 150
        ax.text(bar.get_x() + bar.get_width()/2, height + offset,
               f"{val:,.0f}", ha='center', va='bottom', fontsize=9, fontweight='bold')

    for i, (bar, val) in enumerate(zip(bars2, physical_ops)):
        height = bar.get_height()
        offset = 400 if val > 5000 else 150
        ax.text(bar.get_x() + bar.get_width()/2, height + offset,
               f"{val:,.0f}", ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Reduction percentage annotations - alternating positions to avoid overlap
    speedups = [
        (14463 - 9279) / 14463,  # Total ops reduction: 36.1%
        0,                         # Same allocation
        (14463 - 9279) / (14463),  # Per-frame efficiency: 36.1%
    ]

    for i, (center, speedup) in enumerate(zip(x, speedups)):
        if speedup > 0:
            pct = speedup * 100
            max_val = max(ring_ops[i], physical_ops[i])
            # Alternate positioning: left side for first, right side for last
            if i == 0:  # First bar - place on left side
                ax.text(center - 0.5, max_val + 2500,
                       f"{pct:.1f}%\nreduction", ha='center', va='bottom', fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor=COLOR_SUCCESS, alpha=0.35, pad=0.6))
            else:  # Last bar - place on right side
                ax.text(center + 0.5, max_val + 2500,
                       f"{pct:.1f}%\nreduction", ha='center', va='bottom', fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor=COLOR_SUCCESS, alpha=0.35, pad=0.6))

    ax.set_ylabel("Operation Count", fontsize=12, fontweight='bold')
    ax.set_title("Ring Buffer: CPU Operation Efficiency (8-frame Slide)\nAvoids Expensive memcpy Operations",
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(operations, fontsize=11)
    ax.set_ylim(0, 18000)
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add insight box at the bottom with proper spacing
    fig.text(0.5, 0.03,
            "Ring buffer: circular index avoids memcpy. Saves: memory shifts, malloc/free, GPU-CPU sync",
            ha='center', fontsize=10, style='italic',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.75, pad=0.8, edgecolor='gray', linewidth=1))

    plt.tight_layout()
    plt.savefig("visualizations/output/9_ring_buffer_operation_efficiency.png", dpi=300, bbox_inches='tight')
    plt.savefig("visualizations/output/9_ring_buffer_operation_efficiency.pdf", bbox_inches='tight')
    print("[OK] Saved: 9_ring_buffer_operation_efficiency.png")
    print("[OK] Saved: 9_ring_buffer_operation_efficiency.pdf")
    plt.close()


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("Generating profiling metric visualizations...\n")
    plot_kv_cache_memory_overhead()
    plot_cpu_operations_breakdown()
    plot_cuda_graph_launch_overhead()
    plot_ring_buffer_operation_efficiency()
    print("\nAll profiling metric visualizations generated!")
