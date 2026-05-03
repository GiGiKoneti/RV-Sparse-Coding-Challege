# RV-Sparse LFX Mentorship Coding Challenge

This repository contains the solution for the LFX Mentorship `RV-Sparse` Coding Challenge. The objective is to extract a dense matrix into Compressed Sparse Row (CSR) format and perform Sparse Matrix-Vector Multiplication (SpMV) utilizing caller-provided buffers, with zero dynamic memory allocation.

## Implementation Features

The `sparse_multiply` function is implemented in `challenge.c` with several systems-level optimizations to maximize throughput and minimize latency prior to RISC-V Vector (RVV) integration:

1. **Strict Memory Constraints:** Computations strictly utilize `values`, `col_indices`, and `row_ptrs` buffers. Zero dynamic allocations are performed inside the function.
2. **Alias Analysis (`__restrict`):** All array pointers are explicitly cast with `__restrict` to guarantee non-overlapping memory regions, enabling aggressive compiler instruction reordering and auto-vectorization.
3. **Instruction-Level Parallelism (ILP):** The SpMV inner loop is unrolled 4-ways using multiple independent accumulators (`sum0` through `sum3`), effectively breaking loop-carried dependency chains and allowing the CPU to pipeline floating-point addition instructions.
4. **Software Prefetching:** The gather operation (`x[col_indices[k]]`) induces non-contiguous memory access. A `__builtin_prefetch` call is injected ahead of the loop iteration to fetch future vector indices into the L1 cache, mitigating main memory latency.
5. **Sequential Pointer Arithmetic:** The initial dense matrix scan avoids redundant index multiplication (`i * cols + j`) by utilizing a sequentially incremented pointer (`*rA++`), which perfectly aligns with hardware prefetching heuristics.

## Build and Execute

To compile and run the test harness (recommended to compile with `-O3` to leverage vectorization hints):

```bash
gcc -O3 -lm -o run challenge.c
./run
```