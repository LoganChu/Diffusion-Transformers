"""CUDA-graph-captured ODE solvers for DiT world-model inference.

Captures the entire unrolled ODE loop as a single CUDA graph so that each
inference step = 1 graph replay instead of thousands of individual kernel
launches.

Design constraints (required for CUDA graph capture):
  - All tensor shapes are fixed and pre-allocated as class attributes.
  - No allocations inside the captured region.
  - No Python control flow that depends on GPU tensor values.
  - No .item(), no print, no dynamic shapes.
  - model.prefill_cache() runs OUTSIDE the graph (context changes per step).
  - Caller copies variable inputs (action, initial noise) into pre-allocated
    buffers before calling run(); the graph replays from those buffers.

Classes
-------
GraphedEulerStep
    Captures N-candidate Euler ODE (one horizon step) for CEM planning.
    BS=N=64, num_ode_steps=4 by default.  Works with KVCache or RingKVCache.

GraphedHeunSolver
    Captures full BS=1 Heun ODE for high-quality single-step inference.
    Works with KVCache.

Usage
-----
    from inference.graph_solver import GraphedEulerStep, GraphedHeunSolver

    # CEM inner-loop (one horizon step, N candidates)
    solver = GraphedEulerStep(model, n_ctx=2, N=64, num_ode_steps=4)
    z_next = solver.run(model, ctx_latents_1, a_cond)      # graph replay

    # Single high-quality inference step
    heun   = GraphedHeunSolver(model, n_ctx=2, num_steps=8)
    z_pred = heun.run(model, ctx_latents, ctx_actions, action)  # graph replay
"""

from __future__ import annotations

import torch

from models.cache import KVCache, RingKVCache
from models.dit import (
    ACTION_DIM,
    DEPTH,
    HEAD_DIM,
    IN_CHANNELS,
    LATENT_H,
    LATENT_W,
    NUM_HEADS,
    NUM_PATCHES,
)

_N_WARMUP = 3   # warmup passes before graph capture (stabilises CUDA allocator)


class GraphedEulerStep:
    """CUDA-graph-captured Euler ODE for one CEM horizon step.

    Pre-allocates all GPU buffers and captures the unrolled Euler loop
    (num_ode_steps model evaluations at batch size N) as a single CUDA graph.

    Context K/V is prefilled into a shared BS=1 cache OUTSIDE the graph
    (context changes each horizon step); SDPA broadcasts it to all N candidates
    automatically.  The ODE noise initialisation (x.normal_()) and action copy
    also happen outside the graph so callers can supply deterministic inputs.

    Args:
        model:         DiTSmall in eval mode, float16 on CUDA.
        n_ctx:         Number of context frames (fixes cache size).
        N:             Candidate batch size (default 64, matches CEM default).
        num_ode_steps: Euler steps per horizon step (default 4).
        cache_type:    'kv' for KVCache, 'ring' for RingKVCache.
        dtype:         Tensor dtype (default float16).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        n_ctx: int,
        N: int = 64,
        num_ode_steps: int = 4,
        cache_type: str = "kv",
        dtype: torch.dtype = torch.float16,
    ) -> None:
        device = next(model.parameters()).device
        n_ctx_tokens = n_ctx * NUM_PATCHES

        # --- Pre-allocated static buffers ---
        # x is the denoising latent; written by caller before replay.
        self.x     = torch.empty(N, IN_CHANNELS, LATENT_H, LATENT_W, device=device, dtype=dtype)
        # a holds per-candidate action conditioning; written by caller.
        self.a     = torch.empty(N, ACTION_DIM, device=device, dtype=dtype)
        # t_buf is filled inside the graph with each ODE timestep value.
        self.t_buf = torch.empty(N, device=device, dtype=dtype)

        # Shared BS=1 context cache — prefilled outside graph each horizon step.
        if cache_type == "ring":
            self.cache: KVCache | RingKVCache = RingKVCache(
                DEPTH, NUM_HEADS, HEAD_DIM,
                n_ctx_tokens, NUM_PATCHES, NUM_PATCHES,
                device=device, dtype=dtype,
            )
        else:
            self.cache = KVCache(
                DEPTH, NUM_HEADS, HEAD_DIM,
                n_ctx_tokens, NUM_PATCHES,
                device=device, dtype=dtype,
            )

        self._dt           = 1.0 / num_ode_steps
        self._num_ode_steps = num_ode_steps
        self._graph        = self._capture(model, num_ode_steps, device, dtype)

    def _capture(
        self,
        model: torch.nn.Module,
        num_ode_steps: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.cuda.CUDAGraph:
        dt = self._dt

        def _loop() -> None:
            """Unrolled Euler ODE — captured as CUDA graph."""
            for i in range(num_ode_steps):
                self.t_buf.fill_(i * dt)
                v = model(self.x, self.t_buf, self.a, cache=self.cache)
                self.x.add_(v, alpha=dt)

        # Warmup: run on a side stream to reach steady allocator state.
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(_N_WARMUP):
                _loop()
        torch.cuda.current_stream().wait_stream(side)

        # Capture.
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=side):
            _loop()

        return graph

    @torch.no_grad()
    def run(
        self,
        model: torch.nn.Module,
        ctx_latents_1: torch.Tensor,
        a_cond: torch.Tensor,
        x_init: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Prefill context (outside graph), then replay ODE.

        Args:
            model:         Same model used at construction.
            ctx_latents_1: [1, n_ctx, C, H, W] shared context (all candidates
                           observe the same history — uses BS=1 cache with SDPA
                           broadcast).
            a_cond:        [N, 4] per-candidate action conditioning.
            x_init:        [N, C, H, W] initial noise.  If None, self.x is
                           filled in-place with standard normal noise before
                           replay (useful for benchmarking).

        Returns:
            self.x: [N, C, H, W] predicted next latents (in-place buffer —
                    consume or copy before the next run() call).
        """
        # Context prefill — outside graph; cache state feeds into graph replay.
        model.prefill_cache(ctx_latents_1, a_cond[0:1], self.cache)

        # Copy variable inputs into pre-allocated buffers.
        self.a.copy_(a_cond)
        if x_init is not None:
            self.x.copy_(x_init)
        else:
            self.x.normal_()

        self._graph.replay()
        return self.x


class GraphedHeunSolver:
    """CUDA-graph-captured Heun ODE for high-quality BS=1 inference.

    Captures the full (2*num_steps - 1) model evaluations — Euler predictor +
    corrector for each step — as a single CUDA graph.  Context is prefilled
    outside the graph; the ODE itself (including per-step t_buf fills and the
    in-place Heun accumulation) is entirely inside the graph.

    Args:
        model:     DiTSmall in eval mode, float16 on CUDA.
        n_ctx:     Number of context frames.
        num_steps: Heun ODE steps (default 8 → 15 model evals).
        dtype:     Tensor dtype (default float16).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        n_ctx: int,
        num_steps: int = 8,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        device = next(model.parameters()).device
        n_ctx_tokens = n_ctx * NUM_PATCHES
        B = 1

        # --- Pre-allocated static buffers ---
        self.x      = torch.empty(B, IN_CHANNELS, LATENT_H, LATENT_W, device=device, dtype=dtype)
        # Predictor scratch buffer for Heun corrector step (x + dt*v1).
        self.x_pred = torch.empty(B, IN_CHANNELS, LATENT_H, LATENT_W, device=device, dtype=dtype)
        self.action = torch.empty(B, ACTION_DIM, device=device, dtype=dtype)
        self.t_buf  = torch.empty(B, device=device, dtype=dtype)

        self.cache = KVCache(
            DEPTH, NUM_HEADS, HEAD_DIM,
            n_ctx_tokens, NUM_PATCHES,
            device=device, dtype=dtype,
        )

        self._dt        = 1.0 / num_steps
        self._num_steps = num_steps
        self._graph     = self._capture(model, num_steps, device, dtype)

    def _capture(
        self,
        model: torch.nn.Module,
        num_steps: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.cuda.CUDAGraph:
        dt = self._dt

        def _loop() -> None:
            """Unrolled Heun ODE — captured as CUDA graph.

            Python 'if i < num_steps - 1' is evaluated at capture time and
            baked into the graph as a flat sequence of ops (no runtime branch).
            """
            for i in range(num_steps):
                self.t_buf.fill_(i * dt)
                v1 = model(self.x, self.t_buf, self.action, cache=self.cache)
                if i < num_steps - 1:
                    # Predictor: x_pred = x + dt * v1
                    torch.add(self.x, v1, alpha=dt, out=self.x_pred)
                    self.t_buf.fill_((i + 1) * dt)
                    v2 = model(self.x_pred, self.t_buf, self.action, cache=self.cache)
                    # Corrector: x += dt/2 * (v1 + v2)
                    v1.add_(v2)
                    self.x.add_(v1, alpha=dt * 0.5)
                else:
                    # Final step: plain Euler (no corrector needed)
                    self.x.add_(v1, alpha=dt)

        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(_N_WARMUP):
                _loop()
        torch.cuda.current_stream().wait_stream(side)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=side):
            _loop()

        return graph

    @torch.no_grad()
    def run(
        self,
        model: torch.nn.Module,
        ctx_latents: torch.Tensor,
        ctx_actions: torch.Tensor,
        action: torch.Tensor,
        x_init: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Prefill context (outside graph), then replay Heun ODE.

        Args:
            model:        Same model used at construction.
            ctx_latents:  [1, n_ctx, C, H, W] context frames.
            ctx_actions:  [1, 4] context action conditioning for prefill.
            action:       [1, 4] action conditioning for denoising.
            x_init:       [1, C, H, W] initial noise.  If None, self.x is
                          filled with standard normal noise before replay.

        Returns:
            self.x: [1, C, H, W] predicted latent (in-place buffer).
        """
        model.prefill_cache(ctx_latents, ctx_actions, self.cache)

        self.action.copy_(action)
        if x_init is not None:
            self.x.copy_(x_init)
        else:
            self.x.normal_()

        self._graph.replay()
        return self.x
