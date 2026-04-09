"""
Publication-quality benchmark visualizations for DiT inference optimization.
Run: python visualizations/plot_benchmarks.py

Generates:
1. kv_cache_batch_dependency.png
2. ring_buffer_isolation_vs_reality.png
3. cuda_graph_speedup.png
4. optimization_leaderboard.png
5. amortization_window.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.patches import FancyBboxPatch
import os

os.makedirs("visualizations/output", exist_ok=True)

# Color scheme
COLOR_SUCCESS = "#2ecc71"  # green
COLOR_FAIL = "#e74c3c"     # red
COLOR_NEUTRAL = "#3498db"  # blue
COLOR_ACCENT = "#f39c12"   # orange

# ============================================================================
# Chart 1: KV Cache Speedup vs Batch Size / Eval Count
# ============================================================================

def plot_kv_cache_batch_dependency():
    fig, ax = plt.subplots(figsize=(10, 6))

    scenarios = ["Single Trajectory\n(8 eval)",
                 "Single Trajectory\n(4 eval)",
                 "MPC rollouts\n(256 evals)"]
    speedups = [1.81, 1.63, 0.75]
    colors = [COLOR_SUCCESS, COLOR_SUCCESS, COLOR_FAIL]
    eval_counts = [8, 4, 256]

    # Bar chart
    bars = ax.bar(scenarios, speedups, color=colors, edgecolor='black', linewidth=2, alpha=0.8, width=0.6)

    # Add value labels on bars
    for i, (bar, speedup) in enumerate(zip(bars, speedups)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{speedup:.2f}x',
                ha='center', va='bottom', fontsize=14, fontweight='bold')

    # Breakeven line
    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=2, label='Breakeven')

    # Annotations
    ax.text(0, 1.95, "[+] Cache helps", fontsize=11, ha='center', color=COLOR_SUCCESS, fontweight='bold')
    ax.text(2, 0.55, "[-] Cache hurts", fontsize=11, ha='center', color=COLOR_FAIL, fontweight='bold')

    ax.set_ylabel("Speedup", fontsize=12, fontweight='bold')
    ax.set_xlabel("Eval Count", fontsize=12, fontweight='bold')
    ax.set_title("KV Cache",
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim(0.5, 2.2)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.legend(fontsize=11, loc='upper right')

    plt.tight_layout()
    plt.savefig("visualizations/output/1_kv_cache_batch_dependency.png", dpi=300, bbox_inches='tight')
    print("[OK] Saved: 1_kv_cache_batch_dependency.png")
    plt.close()

# ============================================================================
# Chart 2: Ring-Buffer Isolation vs Reality
# ============================================================================

def plot_ring_buffer_isolation_vs_reality():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Isolated slide
    categories1 = ["Ring\nBuffer", "Sliding\nWindow"]
    times1 = [21.11, 25.94]
    colors1 = [COLOR_SUCCESS, COLOR_NEUTRAL]

    bars1 = ax1.bar(categories1, times1, color=colors1, edgecolor='black', linewidth=2, alpha=0.8, width=0.5)
    ax1.text(0, 21.11 + 1, "21.11 ms", ha='center', fontsize=12, fontweight='bold')
    ax1.text(1, 25.94 + 1, "25.94 ms", ha='center', fontsize=12, fontweight='bold')
    ax1.text(0.5, 10, "1.23x speedup", ha='center', fontsize=13, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor=COLOR_SUCCESS, alpha=0.3))

    ax1.set_ylabel("Total Time (ms)", fontsize=12, fontweight='bold')
    ax1.set_title("Slide Only", fontsize=13, fontweight='bold')
    ax1.set_ylim(0, 35)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    # Right: With inference
    categories2 = ["Ring\nBuffer", "Sliding\nWindow"]
    times2 = [434.31, 436.57]
    colors2 = [COLOR_SUCCESS, COLOR_NEUTRAL]

    bars2 = ax2.bar(categories2, times2, color=colors2, edgecolor='black', linewidth=2, alpha=0.8, width=0.5)
    ax2.text(0, 434.31 + 5, "434.31 ms", ha='center', fontsize=12, fontweight='bold')
    ax2.text(1, 436.57 + 5, "436.57 ms", ha='center', fontsize=12, fontweight='bold')
    ax2.text(0.5, 410, "1.01x speedup\n(noise)", ha='center', fontsize=13, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

    ax2.set_ylabel("Total Time (ms)", fontsize=12, fontweight='bold')
    ax2.set_title("Slide + Inference", fontsize=13, fontweight='bold')
    ax2.set_ylim(0, 500)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    fig.suptitle("Ring-Buffer",
                 fontsize=14, fontweight='bold', y=1.00)

    plt.tight_layout()
    plt.savefig("visualizations/output/2_ring_buffer_isolation_vs_reality.png", dpi=300, bbox_inches='tight')
    print("[OK] Saved: 2_ring_buffer_isolation_vs_reality.png")
    plt.close()

# ============================================================================
# Chart 3: CUDA Graph Speedup
# ============================================================================

def plot_cuda_graph_speedup():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: BS=1, 8 steps
    scenarios1 = ["Eager\nExecution", "CUDA\nGraph"]
    times1 = [118.66, 23.70]
    colors1 = [COLOR_NEUTRAL, COLOR_SUCCESS]

    bars1 = ax1.bar(scenarios1, times1, color=colors1, edgecolor='black', linewidth=2, alpha=0.8, width=0.5)
    ax1.text(0, 118.66 + 5, "118.66 ms", ha='center', fontsize=12, fontweight='bold')
    ax1.text(1, 23.70 + 5, "23.70 ms", ha='center', fontsize=12, fontweight='bold')
    ax1.text(0.5, 80, "5.01x speedup", ha='center', fontsize=14, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor=COLOR_SUCCESS, alpha=0.3))

    ax1.set_ylabel("Total Time (ms)", fontsize=12, fontweight='bold')
    ax1.set_title("Single Trajectory (8 eval)", fontsize=13, fontweight='bold')
    ax1.set_ylim(0, 140)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    # Right: BS=64, 4 steps
    scenarios2 = ["Eager\nExecution", "CUDA\nGraph"]
    times2 = [51.08, 23.33]
    colors2 = [COLOR_NEUTRAL, COLOR_SUCCESS]

    bars2 = ax2.bar(scenarios2, times2, color=colors2, edgecolor='black', linewidth=2, alpha=0.8, width=0.5)
    ax2.text(0, 51.08 + 2, "51.08 ms", ha='center', fontsize=12, fontweight='bold')
    ax2.text(1, 23.33 + 2, "23.33 ms", ha='center', fontsize=12, fontweight='bold')
    ax2.text(0.5, 38, "2.19x speedup", ha='center', fontsize=14, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor=COLOR_SUCCESS, alpha=0.3))

    ax2.set_ylabel("Total Time (ms)", fontsize=12, fontweight='bold')
    ax2.set_title("MPC rollouts (256 evals)", fontsize=13, fontweight='bold')
    ax2.set_ylim(0, 60)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    fig.suptitle("CUDA Graph",
                 fontsize=14, fontweight='bold', y=1.00)

    plt.tight_layout()
    plt.savefig("visualizations/output/3_cuda_graph_speedup.png", dpi=300, bbox_inches='tight')
    print("[OK] Saved: 3_cuda_graph_speedup.png")
    plt.close()

# ============================================================================
# Chart 4: Optimization Leaderboard
# ============================================================================

def plot_optimization_leaderboard():
    fig, ax = plt.subplots(figsize=(12, 8))

    optimizations = [
        "CUDA Graphs - Single Trajectory (8 evals)",
        "CUDA Graphs - MPC Rollouts (256 evals)",
        "KV Cache - Single Trajectory (8 evals)",
        "KV Cache - Single Trajectory (4 evals)",
        "Ring-Buffer Slide",
        "Rolling Inference w/ Ring",
        "KV Cache - MPC Rollouts (256 evals)"
    ]
    speedups = [5.01, 2.19, 1.81, 1.63, 1.23, 1.01, 0.75]
    colors = [COLOR_SUCCESS, COLOR_SUCCESS, COLOR_SUCCESS, COLOR_SUCCESS,
              COLOR_NEUTRAL, COLOR_NEUTRAL, COLOR_FAIL]

    y_pos = np.arange(len(optimizations))

    # Horizontal bars
    bars = ax.barh(y_pos, speedups, color=colors, edgecolor='black', linewidth=2, alpha=0.8, height=0.7)

    # Add value labels
    for i, (bar, speedup) in enumerate(zip(bars, speedups)):
        width = bar.get_width()
        label = f"{speedup:.2f}x"
        if speedup < 1.0:
            label += " (slower)"
        ax.text(width + 0.1, bar.get_y() + bar.get_height()/2, label,
                ha='left', va='center', fontsize=11, fontweight='bold')

    # Breakeven line
    ax.axvline(x=1.0, color='black', linestyle='--', linewidth=2, alpha=0.7, label='Breakeven')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(optimizations, fontsize=11)
    ax.set_xlabel("Speedup (higher is better)", fontsize=12, fontweight='bold')
    ax.set_title("Optimization Leaderboard: Speedup Comparison", fontsize=14, fontweight='bold', pad=20)
    ax.set_xlim(0, 5.5)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.legend(fontsize=10, loc='lower right')

    # Add star for best
    ax.text(5.01 + 0.2, 6, "[BEST]", fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig("visualizations/output/4_optimization_leaderboard.png", dpi=300, bbox_inches='tight')
    print("[OK] Saved: 4_optimization_leaderboard.png")
    plt.close()

# ============================================================================
# Chart 5: Amortization Window Analysis
# ============================================================================

def plot_amortization_window():
    fig, ax = plt.subplots(figsize=(12, 7))

    scenarios = ["Euler BS=1\n8 steps", "Euler BS=1\n4 steps", "MPC BS=64\n4 steps"]
    prefill_cost = [20, 20, 20]  # Approximate
    per_eval_cost = [10, 16.5, 12]
    total_evals = [8, 4, 256]
    speedups = [1.81, 1.63, 0.75]

    x = np.arange(len(scenarios))
    width = 0.35

    # Stacked bars showing prefill + inference cost
    inference_cost = [prefill_cost[i] + per_eval_cost[i] * total_evals[i] for i in range(3)]

    bars1 = ax.bar(x, prefill_cost, width, label='Prefill (one-time)',
                   color=COLOR_ACCENT, edgecolor='black', linewidth=1.5, alpha=0.8)
    bars2 = ax.bar(x, [total_evals[i] * per_eval_cost[i] for i in range(3)], width,
                   bottom=prefill_cost, label='Inference (per-eval)',
                   color=COLOR_NEUTRAL, edgecolor='black', linewidth=1.5, alpha=0.8)

    # Add total time and speedup labels
    for i in range(len(scenarios)):
        total = prefill_cost[i] + total_evals[i] * per_eval_cost[i]
        speedup = speedups[i]
        color = COLOR_SUCCESS if speedup >= 1.6 else (COLOR_FAIL if speedup < 1.2 else COLOR_NEUTRAL)
        ax.text(i, total + 20, f"{speedup:.2f}x", ha='center', fontsize=12, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))
        ax.text(i, -30, f"Amortization:\n{prefill_cost[i]}/{total_evals[i]} evals",
               ha='center', fontsize=10)

    ax.set_ylabel("Total Time (ms, approximate)", fontsize=12, fontweight='bold')
    ax.set_title("Amortization Window: Why Cache Works or Fails", fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, fontsize=11)
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(-50, 500)

    plt.tight_layout()
    plt.savefig("visualizations/output/5_amortization_window.png", dpi=300, bbox_inches='tight')
    print("[OK] Saved: 5_amortization_window.png")
    plt.close()

# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("\nGenerating publication-quality visualizations...\n")

    plot_kv_cache_batch_dependency()
    plot_ring_buffer_isolation_vs_reality()
    plot_cuda_graph_speedup()
    plot_optimization_leaderboard()
    plot_amortization_window()

    print("\nAll visualizations saved to: visualizations/output/")
    print("\nGenerated files:")
    print("  1. 1_kv_cache_batch_dependency.png")
    print("  2. 2_ring_buffer_isolation_vs_reality.png")
    print("  3. 3_cuda_graph_speedup.png")
    print("  4. 4_optimization_leaderboard.png")
    print("  5. 5_amortization_window.png")
    print("\nReady for Google Slides or presentation software.")
