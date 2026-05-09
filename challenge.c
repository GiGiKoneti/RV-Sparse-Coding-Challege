#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// =========================================================
// USER IMPLEMENTATION
// =========================================================
void sparse_multiply(
    int rows, int cols,
    const double* restrict A, const double* restrict x,
    int* restrict out_nnz, double* restrict values,
    int* restrict col_indices, int* restrict row_ptrs,
    double* restrict y
) {
    // Validate inputs defensively (Trap 7 fix)
    if (!A || !x || !values || !col_indices || !row_ptrs || !y || !out_nnz) return;
    if (rows <= 0 || cols <= 0) return;

    int nnz = 0;
    row_ptrs[0] = 0;

    // Pass 1: Extract non-zero elements into CSR format
    for (int i = 0; i < rows; ++i) {
        // Prevent potential int overflow on massive matrices (Trap 6 fix)
        const double* row = &A[(size_t)i * cols];
        for (int j = 0; j < cols; ++j) {
            // Exact comparison is safe: generated values mathematically straddle 0.0 (Trap 14)
            if (row[j] != 0.0) {
                values[nnz] = row[j];
                col_indices[nnz] = j;
                nnz++;
            }
        }
        row_ptrs[i + 1] = nnz;
    }
    *out_nnz = nnz;

    // Pass 2: Compute Sparse Matrix-Vector Multiplication (y = A * x)
    for (int i = 0; i < rows; ++i) {
        int start = row_ptrs[i];
        int end = row_ptrs[i + 1];

        double sum = 0.0;
        for (int k = start; k < end; ++k) {
            sum += values[k] * x[col_indices[k]];
        }
        y[i] = sum;
    }
}

// =========================================================
// TEST HARNESS
// =========================================================
int main(int argc, char* argv[]) {
    // Make tests reproducible (Trap 5 fix)
    unsigned int seed = (argc > 1) ? (unsigned int)atoi(argv[1]) : (unsigned int)time(NULL);
    srand(seed);
    printf("RNG seed: %u (rerun with ./run %u to reproduce)\n\n", seed, seed);

    const int num_iterations = 100;
    int passed_count = 0;

    for (int iter = 0; iter < num_iterations; ++iter) {
        int rows = rand() % 41 + 5;
        int cols = rand() % 41 + 5;
        double density = 0.05 + (rand() / (double)RAND_MAX) * 0.35;

        // Prevent int overflow during size calculation (Trap 6 fix)
        size_t mat_sz = (size_t)rows * cols;

        double* A = calloc(mat_sz, sizeof(double));
        double* values = malloc(mat_sz * sizeof(double));
        int* col_indices = malloc(mat_sz * sizeof(int));
        int* row_ptrs = malloc((rows + 1) * sizeof(int));
        double* x = malloc(cols * sizeof(double));
        
        // Use calloc for output buffer to prevent silent NaN passes (Trap 2 fix)
        double* y_user = calloc(rows, sizeof(double));
        double* y_ref = calloc(rows, sizeof(double));
        int out_nnz = 0;

        // Guard against allocation failures (Trap 4 fix)
        if (!A || !values || !col_indices || !row_ptrs || !x || !y_user || !y_ref) {
            fprintf(stderr, "Allocation failure at iter %d\n", iter);
            free(A); free(values); free(col_indices); free(row_ptrs);
            free(x); free(y_user); free(y_ref);
            return 1;
        }

        for (size_t i = 0; i < mat_sz; ++i) {
            if (((double)rand() / RAND_MAX) < density) {
                A[i] = ((double)rand() / RAND_MAX) * 20.0 - 10.0;
            }
        }

        for (int i = 0; i < cols; ++i) {
            x[i] = ((double)rand() / RAND_MAX) * 20.0 - 10.0;
        }

        // Dense ground truth: y_ref = A * x
        for (int i = 0; i < rows; ++i) {
            double sum = 0.0;
            for (int j = 0; j < cols; ++j) {
                // Prevent int overflow during index calculation (Trap 6 fix)
                sum += A[(size_t)i * cols + j] * x[j];
            }
            y_ref[i] = sum;
        }

        sparse_multiply(rows, cols, A, x, &out_nnz, values, col_indices, row_ptrs, y_user);

        double max_err = 0.0;
        int passed = 1;
        for (int i = 0; i < rows; ++i) {
            double diff = fabs(y_user[i] - y_ref[i]);
            double tol = 1e-7 + 1e-7 * fabs(y_ref[i]);
            
            // Explicitly catch NaN to prevent silent pass vulnerability (Trap 11 fix)
            if (isnan(diff) || diff > tol) {
                if (isnan(diff) || diff > max_err) {
                    max_err = diff; 
                }
                passed = 0;
            }
        }

        if (passed) {
            passed_count++;
            // Don't print max_err on PASS to avoid diagnostic noise (Trap 3 fix)
            printf("Iter %2d [%3dx%3d, density=%.2f, nnz=%4d]: PASS\n",
                   iter, rows, cols, density, out_nnz);
        } else {
            printf("Iter %2d [%3dx%3d, density=%.2f, nnz=%4d]: FAIL (Max error: %.2e)\n",
                   iter, rows, cols, density, out_nnz, max_err);
        }

        free(A);
        free(values);
        free(col_indices);
        free(row_ptrs);
        free(x);
        free(y_user);
        free(y_ref);
    }

    printf("\n%s (%d/%d iterations passed)\n",
           passed_count == num_iterations ? "All tests passed!" : "Some tests failed.",
           passed_count, num_iterations);

    return passed_count == num_iterations ? 0 : 1;
}
