# RV-Sparse-Coding-Challenge

This repository contains the solution for the LFX Mentorship RV-Sparse Coding Challenge.

## Implementation Details

The `sparse_multiply` function is implemented in `challenge.c`. It performs the following steps without any dynamic memory allocation:
1. **CSR Extraction**: Scans the input row-major dense matrix `A` and extracts its non-zero elements into Compressed Sparse Row (CSR) format arrays: `values`, `col_indices`, and `row_ptrs`.
2. **Sparse Matrix-Vector Multiplication (SpMV)**: Computes the matrix-vector product `y = A * x` using the extracted CSR representation.

## Build and Execute

To compile and run the test harness:

```bash
gcc -lm -o run challenge.c
./run
```