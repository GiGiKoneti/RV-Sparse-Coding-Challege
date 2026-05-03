#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// =========================================================
// FUNCTION PROTOTYPE
// =========================================================
void sparse__multiply(
    int rows,
    int cols,
    const double* A,
    const double* x,
    int* out_nnz,
    double* values,
    int* col_indices,
    int* row_ptrs,
    double* y
);

// =========================================================
// USER IMPLEMENTATION
// =========================================================
void sparse_multiply(
    int rows, int cols, const double* A, const double* x,
    int* out_nnz, double* values, int* col_indices, int* row_ptrs,
    double* y
) {
    // Restrict pointers for alias analysis and vectorization hints
    const double* __restrict rA = A;
    const double* __restrict rx = x;
    double* __restrict rvalues = values;
    int* __restrict rcol_indices = col_indices;
    int* __restrict rrow_ptrs = row_ptrs;
    double* __restrict ry = y;

    int nnz = 0;
    rrow_ptrs[0] = 0;
    
    // Pass 1: Extract non-zero elements into CSR format via pointer traversal
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            double val = *rA++;
            if (val != 0.0) {
                rvalues[nnz] = val;
                rcol_indices[nnz] = j;
                nnz++;
            }
        }
        rrow_ptrs[i + 1] = nnz;
    }
    *out_nnz = nnz;

    // Pass 2: Compute Sparse Matrix-Vector Multiplication (SpMV)
    for (int i = 0; i < rows; ++i) {
        int start = rrow_ptrs[i];
        int end = rrow_ptrs[i + 1];
        
        // Break loop-carried dependencies with multiple accumulators (ILP)
        double sum0 = 0.0, sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
        int k = start;
        
        // 4-way unrolling with software prefetching for non-contiguous gather
        for (; k <= end - 4; k += 4) {
            __builtin_prefetch(&rx[rcol_indices[k + 4]], 0, 1);
            
            sum0 += rvalues[k]     * rx[rcol_indices[k]];
            sum1 += rvalues[k + 1] * rx[rcol_indices[k + 1]];
            sum2 += rvalues[k + 2] * rx[rcol_indices[k + 2]];
            sum3 += rvalues[k + 3] * rx[rcol_indices[k + 3]];
        }
        
        double total_sum = (sum0 + sum1) + (sum2 + sum3);
        
        // Remainder loop
        for (; k < end; ++k) {
            total_sum += rvalues[k] * rx[rcol_indices[k]];
        }
        
        ry[i] = total_sum;
    }
}

// =========================================================
// TEST HARNESS
// =========================================================
int main(void) {
    srand(time(NULL));
    
    const int num_iterations = 100;
    int passed_count = 0;

    for (int iter = 0; iter < num_iterations; ++iter) {
        int rows = rand() % 41 + 5;
        int cols = rand() % 41 + 5;
        double density = 0.05 + (rand() / (double) RAND_MAX) * 0.35;
        
        size_t mat_sz = (size_t) rows * cols;

        double* A = calloc(mat_sz, sizeof(double));
        for (size_t i = 0; i < mat_sz; ++i) {
            if (((double) rand() / RAND_MAX) < density) {
                A[i] = ((double) rand() / RAND_MAX) * 20.0 - 10.0;
            }
        }

        double* values = malloc(mat_sz * sizeof(double));
        int* col_indices = malloc(mat_sz * sizeof(int));
        int* row_ptrs = malloc((rows + 1) * sizeof(int));
        double* x = malloc(cols * sizeof(double));
        double* y_user = malloc(rows * sizeof(double));
        double* y_ref = calloc(rows, sizeof(double));
        int out_nnz = 0;

        for (int i = 0; i < cols; ++i) {
            x[i] = ((double) rand() / RAND_MAX) * 20.0 - 10.0;
        }

        for (int i = 0; i < rows; ++i) {
            double sum = 0.0;
            for (int j = 0; j < cols; ++j) {
                sum += A[i * cols + j] * x[j];
            }
            y_ref[i] = sum;
        }

        sparse_multiply(rows, cols, A, x, &out_nnz, values, col_indices, row_ptrs, y_user);

        double max_err = 0.0;
        int passed = 1;
        for (int i = 0; i < rows; ++i) {
            double diff = fabs(y_user[i] - y_ref[i]);
            double tol = 1e-7 + 1e-7 * fabs(y_ref[i]); // Mixed absolute/relative tolerance
            if (diff > tol) {
                max_err = fmax(max_err, diff);
                passed = 0;
            }
        }

        if (passed) {
            passed_count++;
        }

        printf(
            "Iter %2d [%3dx%3d, density=%.2f, nnz=%4d]: %s (Max error: %.2e)\n",
            iter, rows, cols, density, out_nnz, passed ? "PASS" : "FAIL", max_err
        );

        free(A);
        free(values);
        free(col_indices);
        free(row_ptrs);
        free(x);
        free(y_user);
        free(y_ref);
    }

    printf(
        "\n%s (%d/%d iterations passed)\n",
        passed_count == num_iterations ? "All tests passed!" : "Some tests failed.",
        passed_count, num_iterations
    );
           
    return passed_count == num_iterations ? 0 : 1;
}
