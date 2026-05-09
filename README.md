# RV-Sparse LFX Mentorship Coding Challenge

This repository contains the solution for the LFX Mentorship `RV-Sparse` Coding Challenge. The objective is to extract a dense matrix into Compressed Sparse Row (CSR) format and perform Sparse Matrix-Vector Multiplication (SpMV) utilizing caller-provided buffers, with zero dynamic memory allocation.

## Implementation Features

The `sparse_multiply` function is implemented in `challenge.c` with a focus on **correctness, defensive programming, and standard compliance**, while explicitly avoiding premature optimizations that hide structural vulnerabilities:

1. **Strict Memory Constraints:** Computations strictly utilize `values`, `col_indices`, and `row_ptrs` buffers. Zero dynamic allocations are performed inside the function.
2. **Defensive API Design:** The function explicitly validates all input buffers and dimensions to prevent segmentation faults from malformed inputs.
3. **Safe Aliasing via C99 `restrict`:** Array pointers are explicitly annotated with C99 `restrict` (applied to parameters, not locals) to guarantee non-overlapping memory regions, communicating intent safely to the compiler.
4. **Integer Overflow Prevention:** Index calculations (e.g., `i * cols`) are explicitly cast to `size_t` to ensure safety on matrices scaling beyond 2 billion elements, addressing a common vulnerability in sparse linear algebra libraries.
5. **Robust Test Harness Improvements:** The test harness has been fortified against IEEE 754 vulnerabilities (such as silent NaN passes), memory initialization bugs, and non-reproducibility. See `ANALYSIS.md` for a comprehensive breakdown of the test harness audit.

## Build and Execute

To compile and run the test harness:

```bash
gcc -O3 -Wall -Wextra -fsanitize=address -o run challenge.c -lm
./run
```
*(Note: Library linkage `-lm` is placed after the source file to ensure compatibility with Linux GNU ld's default `--as-needed` flag behavior.)*