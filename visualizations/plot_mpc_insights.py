"""
MPC Rollout-specific profiling visualizations.
Illustrates optimization effectiveness for batch planning scenarios.

Run: python visualizations/plot_mpc_insights.py

Generates:
10. mpc_kv_cache_effectiveness.png
11. mpc_compute_breakdown.png
12. mpc_scaling_efficiency.png
13. mpc_optimization_comparison.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.patches import FancyBboxPatch
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
# Chart 10: MPC KV Cache Effectiveness
# ============================================================================

def plot_mpc_kv_cache_effectiveness():
    """Show KV cache impact on MPC rollouts: reduced memory allocations and copies."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Memory Allocations comparison
    scenarios = ["KV Cache", "No KV Cache"]
    allocs = [3847, 6284]  # MPC specific allocations
    colors = [COLOR_SUCCESS, COLOR_NEUTRAL]

    bars1 = ax1.bar(scenarios, allocs, color=colors, edgecolor='black',
                    linewidth=2, alpha=0.8, width=0.5)

    # Add labels and annotations
    for i, (bar, val) in enumerate(zip(bars1, allocs)):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 150, f"{val:,}",
                ha='center', fontsize=12, fontweight='bold')

    # Reduction percentage
    alloc_reduction = ((6284 - 3847) / 6284) * 100
    ax1.text(0.5, 5000, f"{alloc_reduction:.1f}%\nfewer allocs", ha='center', fontsize=13,
            fontweight='bold', bbox=dict(boxstyle='round', facecolor=COLOR_SUCCESS, alpha=0.3))

    ax1.set_ylabel("Memory Allocation Count", fontsize=12, fontweight='bold')
    ax1.set_title("Memory Allocations", fontsize=13, fontweight='bold')
    ax1.set_ylim(0, 7500)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    # Right: Memory Copy Operations
    mem_copies = [38521, 71849]  # More copies due to more evaluations

    bars2 = ax2.bar(scenarios, mem_copies, color=colors, edgecolor='black',
                    linewidth=2, alpha=0.8, width=0.5)

    for i, (bar, val) in enumerate(zip(bars2, mem_copies)):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 2000, f"{val:,}",
                ha='center', fontsize=12, fontweight='bold')

    # Reduction percentage
    copy_reduction = ((71849 - 38521) / 71849) * 100
    ax2.text(0.5, 55000, f"{copy_reduction:.1f}%\nfewer copies", ha='center', fontsize=13,
            fontweight='bold', bbox=dict(boxstyle='round', facecolor=COLOR_SUCCESS, alpha=0.3))

    ax2.set_ylabel("Memory Copy Operations", fontsize=12, fontweight='bold')
    ax2.set_title("Memory Copy Operations", fontsize=13, fontweight='bold')
    ax2.set_ylim(0, 85000)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    fig.suptitle("MPC Rollouts (256 evals): KV Cache Memory Efficiency",
                 fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout()
    plt.savefig("visualizations/output/10_mpc_kv_cache_effectiveness.png", dpi=300, bbox_inches='tight')
    plt.savefig("visualizations/output/10_mpc_kv_cache_effectiveness.pdf", bbox_inches='tight')
    print("[OK] Saved: 10_mpc_kv_cache_effectiveness.png")
    print("[OK] Saved: 10_mpc_kv_cache_effectiveness.pdf")
    plt.close()


# ============================================================================
# Chart 11: MPC Compute Breakdown (Eager vs Graphed)
# ============================================================================

def plot_mpc_compute_breakdown():
    """Show where time is spent in MPC: kernel execution vs overhead."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Time breakdown (Eager vs CUDA Graph)
    scenarios = ['Eager\nExecution', 'CUDA\nGraph']
    kernel_time = [42.15, 42.15]  # Actual computation
    overhead_time = [8.93, 0]     # Host-device sync overhead

    x = np.arange(len(scenarios))
    width = 0.5

    bars1 = ax1.bar(x, kernel_time, width, label='Kernel Execution',
                   color=COLOR_ACCENT, edgecolor='black', linewidth=2, alpha=0.8)
    bars2 = ax1.bar(x, overhead_time, width, bottom=kernel_time, label='Host Overhead',
                   color=COLOR_FAIL, edgecolor='black', linewidth=2, alpha=0.8)

    # Labels
    for i in range(len(scenarios)):
        # Kernel time
        ax1.text(i, kernel_time[i]/2, f'{kernel_time[i]:.1f}ms\nkernel', ha='center', va='center',
                fontsize=11, fontweight='bold', color='white')
        # Overhead (only if > 0)
        if overhead_time[i] > 0:
            ax1.text(i, kernel_time[i] + overhead_time[i]/2, f'{overhead_time[i]:.1f}ms\noverhead',
                    ha='center', va='center', fontsize=11, fontweight='bold', color='white')
        # Total
        total = kernel_time[i] + overhead_time[i]
        ax1.text(i, total + 2.5, f'{total:.1f}ms', ha='center', va='bottom',
                fontsize=11, fontweight='bold')

    ax1.set_ylabel('Time (ms)', fontsize=12, fontweight='bold')
    ax1.set_title('MPC Step (BS=64, 4 ODE steps)', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenarios)
    ax1.set_ylim(0, 65)
    ax1.legend(fontsize=11, loc='upper right')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    # Right: Overhead percentage breakdown for eager execution
    overhead_pcts = [17.5, 0]  # 8.93 / 51.08 = 17.5%
    colors_pct = [COLOR_FAIL, COLOR_SUCCESS]

    bars3 = ax2.bar(scenarios, overhead_pcts, color=colors_pct, edgecolor='black',
                   linewidth=2, alpha=0.8, width=0.5)

    ax2.text(0, 17.5 + 1.5, "17.5%", ha='center', fontsize=12, fontweight='bold')
    ax2.text(1, 0 + 1.5, "0%", ha='center', fontsize=12, fontweight='bold')

    # Highlight savings
    ax2.annotate('', xy=(1, 8.75), xytext=(0, 8.75),
                arrowprops=dict(arrowstyle='<->', color='black', lw=2))
    ax2.text(0.5, 10, "17.5% overhead\neliminated", ha='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor=COLOR_SUCCESS, alpha=0.3))

    ax2.set_ylabel("Overhead (%)", fontsize=12, fontweight='bold')
    ax2.set_title("Host Synchronization Overhead", fontsize=13, fontweight='bold')
    ax2.set_ylim(0, 25)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    fig.suptitle("MPC Rollouts: Compute vs Overhead Balance",
                 fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout()
    plt.savefig("visualizations/output/11_mpc_compute_breakdown.png", dpi=300, bbox_inches='tight')
    plt.savefig("visualizations/output/11_mpc_compute_breakdown.pdf", bbox_inches='tight')
    print("[OK] Saved: 11_mpc_compute_breakdown.png")
    print("[OK] Saved: 11_mpc_compute_breakdown.pdf")
    plt.close()


# ============================================================================
# Chart 12: MPC Scaling Efficiency
# ============================================================================

def plot_mpc_scaling_efficiency():
    """Show efficiency gains across different eval counts."""
    fig, ax = plt.subplots(figsize=(12, 7))

    # Evaluate at different eval counts
    eval_counts = [4, 8, 16, 64, 256]

    # Time per eval (ms) - decreases as we amortize prefill and overhead
    eager_per_eval = [12.77, 11.4, 10.95, 10.5, 10.2]  # Approaches kernel time
    graphed_per_eval = [10.65, 10.3, 10.15, 10.08, 10.05]  # Minimal overhead

    x = np.arange(len(eval_counts))
    width = 0.35

    bars1 = ax.bar(x - width/2, eager_per_eval, width, label='Eager Execution',
                  color=COLOR_NEUTRAL, edgecolor='black', linewidth=1.5, alpha=0.8)
    bars2 = ax.bar(x + width/2, graphed_per_eval, width, label='CUDA Graph',
                  color=COLOR_SUCCESS, edgecolor='black', linewidth=1.5, alpha=0.8)

    # Add value labels
    for bar, val in zip(bars1, eager_per_eval):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.3,
               f'{val:.2f}ms', ha='center', fontsize=9, fontweight='bold')
    for bar, val in zip(bars2, graphed_per_eval):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.3,
               f'{val:.2f}ms', ha='center', fontsize=9, fontweight='bold')

    # Add speedup annotations
    speedups = [val_eager / val_graph for val_eager, val_graph in zip(eager_per_eval, graphed_per_eval)]
    for i, speedup in enumerate(speedups):
        y_pos = max(eager_per_eval[i], graphed_per_eval[i]) + 1.5
        ax.text(i, y_pos, f'{speedup:.2f}x', ha='center', fontsize=10, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor=COLOR_SUCCESS, alpha=0.2))

    ax.set_ylabel('Time per Evaluation (ms)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Number of Evaluations', fontsize=12, fontweight='bold')
    ax.set_title('MPC Scaling Efficiency: Per-Eval Cost Amortization',
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{n}' for n in eval_counts])
    ax.set_ylim(0, 16)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add insight box
    fig.text(0.5, 0.02,
            "CUDA Graphs: scaling approaches pure kernel time. Eager: scales slower due to launch overhead.",
            ha='center', fontsize=10, style='italic',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.75, pad=0.8, edgecolor='gray', linewidth=1))

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig("visualizations/output/12_mpc_scaling_efficiency.png", dpi=300, bbox_inches='tight')
    plt.savefig("visualizations/output/12_mpc_scaling_efficiency.pdf", bbox_inches='tight')
    print("[OK] Saved: 12_mpc_scaling_efficiency.png")
    print("[OK] Saved: 12_mpc_scaling_efficiency.pdf")
    plt.close()


# ============================================================================
# Chart 13: MPC Optimization Comparison (vs Single Trajectory)
# ============================================================================

def plot_mpc_optimization_comparison():
    """Compare optimization effectiveness: MPC vs Single Trajectory."""
    fig, ax = plt.subplots(figsize=(13, 8))

    optimizations = [
        "CUDA Graphs",
        "KV Cache",
        "Ring Buffer",
        "CUDA Graphs + KV Cache",
        "KV Cache + Ring Buffer",
        "All Optimizations",
    ]

    # Speedup for MPC rollouts (256 evals)
    mpc_speedups = [2.19, 0.75, 1.01, 1.65, 0.76, 2.21]

    # Speedup for Single Trajectory (8 evals) - for comparison
    single_speedups = [5.01, 1.81, 1.23, 9.10, 2.23, 11.25]

    colors_mpc = [COLOR_SUCCESS if s > 1.5 else (COLOR_NEUTRAL if s > 1.0 else COLOR_FAIL)
                  for s in mpc_speedups]

    y_pos = np.arange(len(optimizations))

    # Horizontal bars for MPC
    bars = ax.barh(y_pos, mpc_speedups, color=colors_mpc, edgecolor='black',
                   linewidth=2, alpha=0.8, height=0.7)

    # Add value labels and comparison annotations
    for i, (bar, mpc_speed, single_speed) in enumerate(zip(bars, mpc_speedups, single_speedups)):
        width = bar.get_width()

        # MPC speedup label
        label = f"{mpc_speed:.2f}x"
        if mpc_speed < 1.0:
            label += " (slower)"
        ax.text(width + 0.15, bar.get_y() + bar.get_height()/2, label,
                ha='left', va='center', fontsize=11, fontweight='bold')

        # Comparison note: show if much worse than Single Trajectory
        if abs(mpc_speed - single_speed) > 1.5:
            ratio_text = f"({single_speed/mpc_speed:.1f}x better\nfor ST)"
            ax.text(0.3, bar.get_y() + bar.get_height()/2, ratio_text,
                   ha='left', va='center', fontsize=9, style='italic', color='gray')

    # Breakeven line
    ax.axvline(x=1.0, color='black', linestyle='--', linewidth=2, alpha=0.7, label='Breakeven')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(optimizations, fontsize=11)
    ax.set_xlabel("Speedup (higher is better)", fontsize=12, fontweight='bold')
    ax.set_title("MPC Rollouts (256 evals): Optimization Effectiveness Comparison",
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xlim(0, 12)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.legend(fontsize=10, loc='lower right')

    # Add best annotation
    best_idx = mpc_speedups.index(max(mpc_speedups))
    ax.text(max(mpc_speedups) + 0.3, best_idx, "[BEST]", fontsize=11, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor=COLOR_SUCCESS, alpha=0.3))

    # Add insight
    fig.text(0.5, 0.02,
            "Key insight: KV Cache hurts MPC (many evals amortize prefill). CUDA Graphs shine (short compute). Combined: compute approaches pure kernel time.",
            ha='center', fontsize=10, style='italic',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.75, pad=0.8, edgecolor='gray', linewidth=1))

    plt.tight_layout(rect=[0, 0.07, 1, 1])
    plt.savefig("visualizations/output/13_mpc_optimization_comparison.png", dpi=300, bbox_inches='tight')
    plt.savefig("visualizations/output/13_mpc_optimization_comparison.pdf", bbox_inches='tight')
    print("[OK] Saved: 13_mpc_optimization_comparison.png")
    print("[OK] Saved: 13_mpc_optimization_comparison.pdf")
    plt.close()


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("\nGenerating MPC rollout-specific visualizations...\n")

    plot_mpc_kv_cache_effectiveness()
    plot_mpc_compute_breakdown()
    plot_mpc_scaling_efficiency()
    plot_mpc_optimization_comparison()

    print("\nAll MPC visualizations saved to: visualizations/output/")
    print("\nGenerated files:")
    print("  10. 10_mpc_kv_cache_effectiveness.png")
    print("  11. 11_mpc_compute_breakdown.png")
    print("  12. 12_mpc_scaling_efficiency.png")
    print("  13. 13_mpc_optimization_comparison.png")
