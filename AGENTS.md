# OptiWorld-FM: MLSys Engineering Standards

## 1. Performance Mandates
- **Batch Size:** All inference optimizations target Batch Size 1 (BS1).
- **Kernels:** Use `torch.nn.functional.scaled_dot_product_attention` (SDPA) exclusively. No manual attention loops.
- **Precision:** Use `torch.bfloat16` for training and `torch.float16` for inference. Target FP8 (e4m3) for final optimizations on Ada hardware.
- **Memory:** Zero-allocation in the ODE loop. Use `inplace` operations (`add_`, `mul_`) and pre-allocated buffers.

## 2. Structural Requirements
- **KV Caching:** Implement a persistent `KVCache` class that handles "Static Prefix" (historical frames) separately from "Iterative Latents" (ODE steps).
- **CUDA Graphs:** All sampling loops must be "Graph-Safe": No `.item()`, no `if/else` on GPU tensors, no `print`, and no dynamic shapes.
- **Profiling:** Every module must use `torch.cuda.nvtx.range_push/pop` or `torch.profiler.record_function` to label blocks for Nsight Systems.

## 3. Tooling & Commands
- **Profile:** `nsys profile -w true -t cuda,nvtx,osrt -o profiler/rep_%b python inference/bench.py`
- **Check Memory:** `nvidia-smi --query-gpu=memory.used --format=csv`
- **Compile:** Always test with `torch.compile(model, mode="max-autotune")`.

## 4. Environment Setup
- **Conda:** `Always use the ml_env conda environment when running things`