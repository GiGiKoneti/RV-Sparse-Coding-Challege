# 🚀 RV-Sparse LFX Mentorship Coding Challenge

![C](https://img.shields.io/badge/Language-C99-blue.svg)
![Build](https://img.shields.io/badge/Build-Passing-brightgreen.svg)
![Memory](https://img.shields.io/badge/Zero_Allocations-Verified-success.svg)

This repository contains my official submission for the LFX Mentorship **RV-Sparse** Coding Challenge (Summer 2026). 

The objective of this challenge is to extract a dense matrix into Compressed Sparse Row (CSR) format and perform Sparse Matrix-Vector Multiplication (SpMV) utilizing caller-provided buffers, strictly enforcing a **zero dynamic memory allocation** constraint.

---

## 🔬 Beyond the Implementation: Template Analysis

During the implementation process, I performed a deep numerical and systems-level audit of the provided `challenge.c` test harness. 

I discovered **28 distinct vulnerabilities and edge cases** intentionally (and perhaps unintentionally) built into the template, including a critical IEEE 754 `NaN` silent-pass exploit and an unenforced CSR validation bypass.

👉 **Please read the full breakdown here:** [**`ANALYSIS.md`**](./ANALYSIS.md)

---

## 🛠️ Implementation Features

The `sparse_multiply` function is implemented in `challenge.c` with a focus on **correctness, defensive programming, and standard compliance**. I explicitly avoided premature optimizations (like unrolling) that lack measurable benefit at the current test sizes, focusing instead on structural integrity:

1. **Strict Memory Constraints:** Computations strictly utilize `values`, `col_indices`, and `row_ptrs` buffers. Zero dynamic allocations are performed inside the function.
2. **Defensive API Design:** The function explicitly validates all input buffers and dimensions to prevent segmentation faults from malformed inputs.
3. **Safe Aliasing via C99 `restrict`:** Array pointers are explicitly annotated with C99 `restrict` (applied to parameters, not locals) to guarantee non-overlapping memory regions, communicating intent safely to the compiler.
4. **Integer Overflow Prevention:** Index calculations (e.g., `i * cols`) are explicitly cast to `size_t` to ensure safety on matrices scaling beyond 2 billion elements.
5. **Robust Test Harness Improvements:** The test harness has been fortified against IEEE 754 vulnerabilities, memory initialization bugs (`malloc` vs `calloc`), and non-reproducibility.

---

## 🚀 Build and Execute

To compile and run the test harness with full safety checks:

```bash
# Compile with strict warnings and AddressSanitizer
gcc -O3 -Wall -Wextra -Wpedantic -fsanitize=address -o run challenge.c -lm

# Execute
./run
```

*(Note: Library linkage `-lm` is intentionally placed **after** the source file to ensure compatibility with Linux GNU `ld`'s default `--as-needed` flag behavior, preventing linker errors.)*