"""Comprehensive profiling script for DiT inference benchmarks via torch.profiler.

Uses workload factories from inference/bench.py to ensure single source of truth
for benchmark logic. Outputs TensorBoard-compatible .pt.trace.json files.

Run in Google Colab with:
    !pip install torch-tb-profiler
    %run profiler/profile_colab.py
    %load_ext tensorboard
    %tensorboard --logdir /content/traces

Or locally:
    python profiler/profile_colab.py
"""

import sys
import os

# --- Google Drive mount + sys.path setup ---
try:
    from google.colab import drive
    drive.mount("/content/drive")
    REPO_ROOT = "/content/Diffusion-Transformers"  # Adjust to your actual path
except ImportError:
    # Not in Colab — assume script is run from repo root directly
    REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

CKPT_PATH = os.path.join(REPO_ROOT, "offline_best.pt")
TRACE_DIR = "/content/traces" if os.path.exists("/content") else "/tmp/traces"
os.makedirs(TRACE_DIR, exist_ok=True)

# --- Core imports ---
import torch
from torch.profiler import profile, ProfilerActivity, record_function, schedule

from models.dit import (
    ACTION_DIM, DEPTH, HEAD_DIM, IN_CHANNELS,
    LATENT_H, LATENT_W, NUM_HEADS, NUM_PATCHES, DiTSmall,
)
from inference.bench import (
    make_workload_euler_bs1_cached,
    make_workload_euler_bs1_cached_ring,
    make_workload_euler_bs1_recompute,
    make_workload_mpc_recompute,
    make_workload_mpc_cached,
    make_workload_ungraphed_euler_step,
    make_workload_graphed_euler_step,
    make_workload_graphed_euler_bs1,
    make_workload_slide_physical,
    make_workload_slide_ring,
)
from inference.graph_solver import GraphedEulerStep, GraphedHeunSolver

# --- Profiler configuration ---
PROF_SCHEDULE = schedule(skip_first=2, wait=1, warmup=1, active=3, repeat=1)
# Total steps needed per benchmark: 2+1+1+3 = 7


def load_model(ckpt_path: str, device: torch.device) -> DiTSmall:
    """Load the DiTSmall model from checkpoint."""
    model = DiTSmall().to(device=device, dtype=torch.float16).eval()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state)
    return model


def make_inputs(n_ctx: int = 2, B: int = 1, N: int = 64,
                device: torch.device = torch.device("cuda"),
                dtype: torch.dtype = torch.float16):
    """Generate synthetic inputs for benchmarks."""
    g = torch.Generator(device=device).manual_seed(42)
    return dict(
        ctx_latents  = torch.randn(B, n_ctx, IN_CHANNELS, LATENT_H, LATENT_W,
                                   device=device, dtype=dtype, generator=g),
        ctx_actions  = torch.randn(B, ACTION_DIM, device=device, dtype=dtype, generator=g),
        action       = torch.randn(B, ACTION_DIM, device=device, dtype=dtype, generator=g),
        a_cond_N     = torch.randn(N, ACTION_DIM, device=device, dtype=dtype, generator=g),
        x_init_bs1   = torch.randn(B, IN_CHANNELS, LATENT_H, LATENT_W,
                                   device=device, dtype=dtype, generator=g),
        x_init_N     = torch.randn(N, IN_CHANNELS, LATENT_H, LATENT_W,
                                   device=device, dtype=dtype, generator=g),
    )


def run_profiler(label: str, workload_fn, n_steps: int = 7):
    """
    Run workload_fn under torch.profiler for n_steps calls.
    Exports TensorBoard .pt.trace.json and prints key_averages table.

    Args:
        label: Name for the trace file and run.
        workload_fn: Callable with no arguments.
        n_steps: Total profiler steps (default 7 = skip_first + wait + warmup + active).
    """
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=PROF_SCHEDULE,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(TRACE_DIR,
                                                                 worker_name=label),
        with_stack=True,
        record_shapes=True,
        profile_memory=False,   # Keep trace file size manageable on T4
    ) as prof:
        for _ in range(n_steps):
            workload_fn()
            prof.step()

    # Print table summary
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    print(prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=20,
    ))


# --- Benchmark 1: Euler BS=1 (KV / Ring / Recompute) ---
def profile_euler_bs1(model, inputs, num_steps: int = 8):
    """Profile Euler BS=1 with different cache types."""
    ctx = inputs["ctx_latents"]
    ctx_a = inputs["ctx_actions"]
    action = inputs["action"]

    def euler_kv():
        with record_function("bench.euler_bs1_kv"):
            workload = make_workload_euler_bs1_cached(model, ctx, ctx_a, action, num_steps)
            workload()

    def euler_ring():
        with record_function("bench.euler_bs1_ring"):
            workload = make_workload_euler_bs1_cached_ring(model, ctx, ctx_a, action, num_steps)
            workload()

    def euler_recompute():
        with record_function("bench.euler_bs1_recompute"):
            workload = make_workload_euler_bs1_recompute(model, ctx, ctx_a, action, num_steps)
            workload()

    run_profiler("euler_bs1_kv",        euler_kv)
    run_profiler("euler_bs1_ring",      euler_ring)
    run_profiler("euler_bs1_recompute", euler_recompute)


# --- Benchmark 2: MPC step (KV / Ring / Recompute) ---
def profile_mpc_step(model, inputs, N: int = 64, num_ode_steps: int = 4):
    """Profile MPC single step with different cache types."""
    ctx = inputs["ctx_latents"]

    def mpc_kv():
        with record_function("bench.mpc_kv"):
            workload = make_workload_mpc_cached(model, ctx, N, num_ode_steps, cache_type="kv")
            workload()

    def mpc_ring():
        with record_function("bench.mpc_ring"):
            workload = make_workload_mpc_cached(model, ctx, N, num_ode_steps, cache_type="ring")
            workload()

    def mpc_recompute():
        with record_function("bench.mpc_recompute"):
            workload = make_workload_mpc_recompute(model, ctx, N, num_ode_steps)
            workload()

    run_profiler("mpc_kv",        mpc_kv)
    run_profiler("mpc_ring",      mpc_ring)
    run_profiler("mpc_recompute", mpc_recompute)


# --- Benchmark 3: Slide (Physical vs Ring) ---
def profile_slide(model, inputs, n_roll_frames: int = 8):
    """Profile KVCache.slide vs RingKVCache.slide_ring."""
    ctx = inputs["ctx_latents"]
    ctx_a = inputs["ctx_actions"]
    action = inputs["action"]
    device = ctx.device
    dtype = next(model.parameters()).dtype
    n_frame_shape = (1, NUM_HEADS, NUM_PATCHES, HEAD_DIM)

    # Pre-compute new-frame K/V (same for both benchmarks — eliminates noise)
    g = torch.Generator(device=device).manual_seed(0)
    new_frame_kvs = [
        (
            torch.randn(*n_frame_shape, device=device, dtype=dtype, generator=g),
            torch.randn(*n_frame_shape, device=device, dtype=dtype, generator=g),
        )
        for _ in range(n_roll_frames)
    ]

    def slide_physical():
        with record_function("bench.slide_physical"):
            workload = make_workload_slide_physical(model, ctx, ctx_a, n_roll_frames, new_frame_kvs)
            workload()

    def slide_ring():
        with record_function("bench.slide_ring"):
            workload = make_workload_slide_ring(model, ctx, ctx_a, n_roll_frames, new_frame_kvs)
            workload()

    run_profiler("slide_physical", slide_physical)
    run_profiler("slide_ring",     slide_ring)


# --- Benchmark 4: Graphed vs Eager ---
def profile_graphed_vs_eager(model, inputs, n_ctx, N: int = 64, num_ode_steps: int = 4,
                             num_steps_bs1: int = 8):
    """Profile CUDA graphed vs eager execution."""
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    ctx = inputs["ctx_latents"]
    ctx_a = inputs["ctx_actions"]
    action = inputs["action"]

    # Construct solvers outside profiling (one-time capture cost)
    print("\nConstructing CUDA graphs (this may take 10-30 seconds)...")
    solver_euler_kv   = GraphedEulerStep(model, n_ctx=n_ctx, N=N,
                                          num_ode_steps=num_ode_steps,
                                          cache_type="kv", dtype=dtype)
    solver_euler_ring = GraphedEulerStep(model, n_ctx=n_ctx, N=N,
                                          num_ode_steps=num_ode_steps,
                                          cache_type="ring", dtype=dtype)
    solver_heun_kv    = GraphedHeunSolver(model, n_ctx=n_ctx,
                                           num_steps=num_steps_bs1,
                                           cache_type="kv", dtype=dtype)
    solver_heun_ring  = GraphedHeunSolver(model, n_ctx=n_ctx,
                                           num_steps=num_steps_bs1,
                                           cache_type="ring", dtype=dtype)
    print("Graphs constructed.")

    # Eager MPC (N candidates, KV-cached)
    def eager_mpc_kv():
        with record_function("bench.eager_mpc_kv"):
            workload = make_workload_mpc_cached(model, ctx, N, num_ode_steps, cache_type="kv")
            workload()

    def graphed_euler_kv():
        with record_function("bench.graphed_euler_kv"):
            workload = make_workload_graphed_euler_step(solver_euler_kv, model, ctx, N)
            workload()

    def graphed_euler_ring():
        with record_function("bench.graphed_euler_ring"):
            workload = make_workload_graphed_euler_step(solver_euler_ring, model, ctx, N)
            workload()

    def graphed_heun_kv():
        with record_function("bench.graphed_heun_kv"):
            workload = make_workload_graphed_euler_bs1(solver_heun_kv, model, ctx, ctx_a, action)
            workload()

    def graphed_heun_ring():
        with record_function("bench.graphed_heun_ring"):
            workload = make_workload_graphed_euler_bs1(solver_heun_ring, model, ctx, ctx_a, action)
            workload()

    run_profiler("graphed_euler_kv",   graphed_euler_kv)
    run_profiler("graphed_euler_ring", graphed_euler_ring)
    run_profiler("graphed_heun_kv",    graphed_heun_kv)
    run_profiler("graphed_heun_ring",  graphed_heun_ring)
    run_profiler("eager_mpc_kv",       eager_mpc_kv)


# --- Benchmark 5: Ungraphed Euler ---
def profile_ungraphed(model, inputs, N: int = 64, num_ode_steps: int = 4):
    """Profile ungraphed Euler step baseline."""
    ctx = inputs["ctx_latents"]

    def ungraphed_euler():
        with record_function("bench.ungraphed_euler"):
            workload = make_workload_ungraphed_euler_step(model, ctx, N, num_ode_steps)
            workload()

    run_profiler("ungraphed_euler", ungraphed_euler)


def main():
    """Run all profiling benchmarks."""
    device = torch.device("cuda")
    dtype  = torch.float16
    N_CTX  = 2
    N      = 64
    NUM_ODE_STEPS = 4
    NUM_STEPS_BS1 = 8
    N_ROLL_FRAMES = 8

    print("Loading model...")
    model = load_model(CKPT_PATH, device)
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

    inputs = make_inputs(n_ctx=N_CTX, B=1, N=N, device=device, dtype=dtype)

    print("\n" + "="*70)
    print("  Benchmark 1: Euler BS=1 (KV / Ring / Recompute)")
    print("="*70)
    profile_euler_bs1(model, inputs, num_steps=NUM_STEPS_BS1)

    print("\n" + "="*70)
    print("  Benchmark 2: MPC Step (KV / Ring / Recompute)")
    print("="*70)
    profile_mpc_step(model, inputs, N=N, num_ode_steps=NUM_ODE_STEPS)

    print("\n" + "="*70)
    print("  Benchmark 3: Slide (Physical vs Ring)")
    print("="*70)
    profile_slide(model, inputs, n_roll_frames=N_ROLL_FRAMES)

    print("\n" + "="*70)
    print("  Benchmark 4: Graphed vs Eager")
    print("="*70)
    profile_graphed_vs_eager(model, inputs, N_CTX, N=N, num_ode_steps=NUM_ODE_STEPS,
                             num_steps_bs1=NUM_STEPS_BS1)

    print("\n" + "="*70)
    print("  Benchmark 5: Ungraphed Euler (baseline)")
    print("="*70)
    profile_ungraphed(model, inputs, N=N, num_ode_steps=NUM_ODE_STEPS)

    print("\n" + "="*70)
    print(f"All traces written to: {TRACE_DIR}")
    print("="*70)
    print("\nTo download traces (Colab):")
    print("  from google.colab import files")
    print("  files.download('/content/traces')")
    print("\nTo view in TensorBoard:")
    print("  %tensorboard --logdir /content/traces")


if __name__ == "__main__":
    main()
