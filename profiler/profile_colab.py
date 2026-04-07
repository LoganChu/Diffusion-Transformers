"""Comprehensive profiling script for DiT inference benchmarks.

Run in Google Colab with:
    !pip install torch-tb-profiler
    %run profiler/profile_colab.py
    %load_ext tensorboard
    %tensorboard --logdir /content/traces

Outputs Chrome trace JSON and TensorBoard .pt.trace.json files to /content/traces/.
"""

import sys
import os

# --- Google Drive mount + sys.path setup ---
try:
    from google.colab import drive
    drive.mount("/content/drive")
    REPO_ROOT = "/content/drive/MyDrive/Diffusion-Transformers"  # Adjust to your actual path
except ImportError:
    # Not in Colab — assume script is run from repo root directly
    REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

CKPT_PATH  = os.path.join(REPO_ROOT, "offline_best.pt")
TRACE_DIR  = "/content/traces"
os.makedirs(TRACE_DIR, exist_ok=True)

# --- Core imports ---
import torch
import torch.cuda.nvtx as nvtx
from torch.profiler import profile, ProfilerActivity, record_function, schedule

from models.cache import KVCache, RingKVCache
from models.dit import (
    ACTION_DIM, DEPTH, HEAD_DIM, IN_CHANNELS,
    LATENT_H, LATENT_W, NUM_HEADS, NUM_PATCHES, DiTSmall,
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


def _make_cache(n_ctx_tokens: int, device, dtype, cache_type: str = "kv"):
    """Factory for KVCache or RingKVCache."""
    if cache_type == "ring":
        return RingKVCache(
            DEPTH, NUM_HEADS, HEAD_DIM,
            n_ctx_tokens, NUM_PATCHES, NUM_PATCHES,
            device=device, dtype=dtype,
        )
    return KVCache(
        DEPTH, NUM_HEADS, HEAD_DIM,
        n_ctx_tokens, NUM_PATCHES,
        device=device, dtype=dtype,
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
    trace_path = os.path.join(TRACE_DIR, f"{label}.json")

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
    print(f"Trace saved to: {trace_path}")


# --- Benchmark 1: Euler BS=1 (KV / Ring / Recompute) ---
def profile_euler_bs1(model, inputs, n_ctx_tokens, device, dtype,
                       num_steps: int = 8):
    """Profile Euler BS=1 with different cache types."""
    dt = 1.0 / num_steps
    t_buf = torch.empty(1, device=device, dtype=dtype)
    ctx = inputs["ctx_latents"]
    ctx_a = inputs["ctx_actions"]
    action = inputs["action"]

    def euler_kv():
        with torch.amp.autocast("cuda", dtype=torch.float16):
            with record_function("bench.euler_bs1_kv"):
                cache = _make_cache(n_ctx_tokens, device, dtype, "kv")
                model.prefill_cache(ctx, ctx_a, cache)
                x = inputs["x_init_bs1"].clone()
                for i in range(num_steps):
                    t_buf.fill_(i * dt)
                    v = model(x, t_buf, action, cache=cache)
                    x.add_(v, alpha=dt)

    def euler_ring():
        with torch.amp.autocast("cuda", dtype=torch.float16):
            with record_function("bench.euler_bs1_ring"):
                cache = _make_cache(n_ctx_tokens, device, dtype, "ring")
                model.prefill_cache(ctx, ctx_a, cache)
                x = inputs["x_init_bs1"].clone()
                for i in range(num_steps):
                    t_buf.fill_(i * dt)
                    v = model(x, t_buf, action, cache=cache)
                    x.add_(v, alpha=dt)

    def euler_recompute():
        with torch.amp.autocast("cuda", dtype=torch.float16):
            with record_function("bench.euler_bs1_recompute"):
                x = inputs["x_init_bs1"].clone()
                for i in range(num_steps):
                    cache = _make_cache(n_ctx_tokens, device, dtype, "kv")
                    model.prefill_cache(ctx, ctx_a, cache)
                    t_val = torch.full((1,), i * dt, device=device, dtype=dtype)
                    v = model(x, t_val, action, cache=cache)
                    x = x + dt * v

    run_profiler("euler_bs1_kv",        euler_kv)
    run_profiler("euler_bs1_ring",      euler_ring)
    run_profiler("euler_bs1_recompute", euler_recompute)


# --- Benchmark 2: MPC step (KV / Ring / Recompute) ---
def profile_mpc_step(model, inputs, n_ctx_tokens, device, dtype,
                      N: int = 64, num_ode_steps: int = 4):
    """Profile MPC single step with different cache types."""
    dt = 1.0 / num_ode_steps
    ctx = inputs["ctx_latents"]
    ctx_a = inputs["ctx_actions"]
    a_cond = inputs["a_cond_N"]

    def mpc_kv():
        with torch.amp.autocast("cuda", dtype=torch.float16):
            with record_function("bench.mpc_kv"):
                cache = _make_cache(n_ctx_tokens, device, dtype, "kv")
                model.prefill_cache(ctx[0:1], a_cond[0:1], cache)
                x = inputs["x_init_N"].clone()
                t_buf = torch.empty(N, device=device, dtype=dtype)
                for i in range(num_ode_steps):
                    t_buf.fill_(i * dt)
                    v = model(x, t_buf, a_cond, cache=cache)
                    x.add_(v, alpha=dt)

    def mpc_ring():
        with torch.amp.autocast("cuda", dtype=torch.float16):
            with record_function("bench.mpc_ring"):
                cache = _make_cache(n_ctx_tokens, device, dtype, "ring")
                model.prefill_cache(ctx[0:1], a_cond[0:1], cache)
                x = inputs["x_init_N"].clone()
                t_buf = torch.empty(N, device=device, dtype=dtype)
                for i in range(num_ode_steps):
                    t_buf.fill_(i * dt)
                    v = model(x, t_buf, a_cond, cache=cache)
                    x.add_(v, alpha=dt)

    def mpc_recompute():
        with torch.amp.autocast("cuda", dtype=torch.float16):
            with record_function("bench.mpc_recompute"):
                ctx_N = ctx.expand(N, -1, -1, -1, -1)
                x = inputs["x_init_N"].clone()
                for i in range(num_ode_steps):
                    cache = _make_cache(n_ctx_tokens, device, dtype, "kv")
                    model.prefill_cache(ctx_N, a_cond, cache)
                    t_val = torch.full((N,), i * dt, device=device, dtype=dtype)
                    v = model(x, t_val, a_cond, cache=cache)
                    x = x + dt * v

    run_profiler("mpc_kv",        mpc_kv)
    run_profiler("mpc_ring",      mpc_ring)
    run_profiler("mpc_recompute", mpc_recompute)


# --- Benchmark 3: Slide (Physical vs Ring) ---
def profile_slide(model, inputs, n_ctx_tokens, device, dtype,
                   n_roll_frames: int = 8):
    """Profile KVCache.slide vs RingKVCache.slide_ring."""
    ctx = inputs["ctx_latents"]
    ctx_a = inputs["ctx_actions"]
    n_frame_shape = (1, NUM_HEADS, NUM_PATCHES, HEAD_DIM)
    g = torch.Generator(device=device).manual_seed(0)
    new_frame_kvs = [
        (
            torch.randn(*n_frame_shape, device=device, dtype=dtype, generator=g),
            torch.randn(*n_frame_shape, device=device, dtype=dtype, generator=g),
        )
        for _ in range(n_roll_frames)
    ]

    def slide_physical():
        with torch.amp.autocast("cuda", dtype=torch.float16):
            with record_function("bench.slide_physical"):
                cache = _make_cache(n_ctx_tokens, device, dtype, "kv")
                model.prefill_cache(ctx, ctx_a, cache)
                for frame_idx in range(n_roll_frames):
                    k_new, v_new = new_frame_kvs[frame_idx]
                    for layer_idx in range(DEPTH):
                        cache.slide(layer_idx, k_new, v_new)

    def slide_ring():
        with torch.amp.autocast("cuda", dtype=torch.float16):
            with record_function("bench.slide_ring"):
                cache = _make_cache(n_ctx_tokens, device, dtype, "ring")
                model.prefill_cache(ctx, ctx_a, cache)
                for frame_idx in range(n_roll_frames):
                    k_new, v_new = new_frame_kvs[frame_idx]
                    for layer_idx in range(DEPTH):
                        cache.slide_ring(layer_idx, k_new, v_new)
                    cache.advance_head()

    run_profiler("slide_physical", slide_physical)
    run_profiler("slide_ring",     slide_ring)


# --- Benchmark 4: Graphed vs Eager ---
def profile_graphed_vs_eager(model, inputs, n_ctx, device, dtype,
                               N: int = 64, num_ode_steps: int = 4,
                               num_steps_bs1: int = 8):
    """Profile CUDA graphed vs eager execution."""
    ctx = inputs["ctx_latents"]
    ctx_a = inputs["ctx_actions"]
    action = inputs["action"]
    a_cond = inputs["a_cond_N"]

    # Construct solvers outside profiling (one-time capture cost)
    print("\nConstructing CUDA graphs (this may take 10-30 seconds)...")
    solver_euler_kv   = GraphedEulerStep(model, n_ctx=n_ctx, N=N,
                                          num_ode_steps=num_ode_steps,
                                          cache_type="kv",   dtype=dtype)
    solver_euler_ring = GraphedEulerStep(model, n_ctx=n_ctx, N=N,
                                          num_ode_steps=num_ode_steps,
                                          cache_type="ring", dtype=dtype)
    solver_heun_kv    = GraphedHeunSolver(model, n_ctx=n_ctx,
                                           num_steps=num_steps_bs1,
                                           cache_type="kv",  dtype=dtype)
    solver_heun_ring  = GraphedHeunSolver(model, n_ctx=n_ctx,
                                           num_steps=num_steps_bs1,
                                           cache_type="ring", dtype=dtype)
    print("Graphs constructed.")

    # Eager MPC (N candidates, KV-cached)
    dt = 1.0 / num_ode_steps
    n_ctx_tokens = n_ctx * NUM_PATCHES

    def eager_mpc_kv():
        with torch.amp.autocast("cuda", dtype=torch.float16):
            with record_function("bench.eager_mpc_kv"):
                cache = _make_cache(n_ctx_tokens, device, dtype, "kv")
                model.prefill_cache(ctx[0:1], a_cond[0:1], cache)
                x = inputs["x_init_N"].clone()
                t_buf = torch.empty(N, device=device, dtype=dtype)
                for i in range(num_ode_steps):
                    t_buf.fill_(i * dt)
                    v = model(x, t_buf, a_cond, cache=cache)
                    x.add_(v, alpha=dt)

    def graphed_euler_kv():
        with record_function("bench.graphed_euler_kv"):
            solver_euler_kv.run(model, ctx, a_cond,
                                 x_init=inputs["x_init_N"])

    def graphed_euler_ring():
        with record_function("bench.graphed_euler_ring"):
            solver_euler_ring.run(model, ctx, a_cond,
                                   x_init=inputs["x_init_N"])

    def graphed_heun_kv():
        with record_function("bench.graphed_heun_kv"):
            solver_heun_kv.run(model, ctx, ctx_a, action,
                                x_init=inputs["x_init_bs1"])

    def graphed_heun_ring():
        with record_function("bench.graphed_heun_ring"):
            solver_heun_ring.run(model, ctx, ctx_a, action,
                                  x_init=inputs["x_init_bs1"])

    run_profiler("graphed_euler_kv",   graphed_euler_kv)
    run_profiler("graphed_euler_ring", graphed_euler_ring)
    run_profiler("graphed_heun_kv",    graphed_heun_kv)
    run_profiler("graphed_heun_ring",  graphed_heun_ring)
    run_profiler("eager_mpc_kv",       eager_mpc_kv)


# --- Benchmark 5: Prefill cost ---
def profile_prefill(model, inputs, device, dtype):
    """Profile context encoding (prefill) cost."""
    ctx = inputs["ctx_latents"]
    ctx_a = inputs["ctx_actions"]
    n_ctx_tokens = ctx.shape[1] * NUM_PATCHES

    def prefill_kv():
        with torch.amp.autocast("cuda", dtype=torch.float16):
            with record_function("bench.prefill_kv"):
                cache = _make_cache(n_ctx_tokens, device, dtype, "kv")
                model.prefill_cache(ctx, ctx_a, cache)

    def prefill_ring():
        with torch.amp.autocast("cuda", dtype=torch.float16):
            with record_function("bench.prefill_ring"):
                cache = _make_cache(n_ctx_tokens, device, dtype, "ring")
                model.prefill_cache(ctx, ctx_a, cache)

    run_profiler("prefill_kv",   prefill_kv)
    run_profiler("prefill_ring", prefill_ring)


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

    n_ctx_tokens = N_CTX * NUM_PATCHES
    inputs = make_inputs(n_ctx=N_CTX, B=1, N=N, device=device, dtype=dtype)

    print("\n" + "="*70)
    print("  Benchmark 1: Euler BS=1 (KV / Ring / Recompute)")
    print("="*70)
    profile_euler_bs1(model, inputs, n_ctx_tokens, device, dtype,
                       num_steps=NUM_STEPS_BS1)

    print("\n" + "="*70)
    print("  Benchmark 2: MPC Step (KV / Ring / Recompute)")
    print("="*70)
    profile_mpc_step(model, inputs, n_ctx_tokens, device, dtype,
                      N=N, num_ode_steps=NUM_ODE_STEPS)

    print("\n" + "="*70)
    print("  Benchmark 3: Slide (Physical vs Ring)")
    print("="*70)
    profile_slide(model, inputs, n_ctx_tokens, device, dtype,
                   n_roll_frames=N_ROLL_FRAMES)

    print("\n" + "="*70)
    print("  Benchmark 4: Graphed vs Eager")
    print("="*70)
    profile_graphed_vs_eager(model, inputs, N_CTX, device, dtype,
                              N=N, num_ode_steps=NUM_ODE_STEPS,
                              num_steps_bs1=NUM_STEPS_BS1)

    print("\n" + "="*70)
    print("  Benchmark 5: Prefill Cost")
    print("="*70)
    profile_prefill(model, inputs, device, dtype)

    print("\n" + "="*70)
    print("  PROFILING COMPLETE")
    print("="*70)
    print(f"\nAll traces written to: {TRACE_DIR}")
    print("\nIn Colab, view traces with TensorBoard:")
    print("  %load_ext tensorboard")
    print("  %tensorboard --logdir /content/traces")
    print("\nOr download individual JSON files from the Files panel (left sidebar).")


if __name__ == "__main__":
    main()
