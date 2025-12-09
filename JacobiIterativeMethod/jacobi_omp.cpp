#include <iostream>
#include <vector>
#include <cmath>
#include <time.h>
#include <omp.h>
#include "jacobi_omp.h"

using namespace std;

void jacobi_omp(vector<vector<double>>& A, vector<double>& b,
    int n, int maxIter, double tol, int numThreads)
{
    // Step 1: Initialize guess vectors
    vector<double> x(n, 0.0), x_new(n, 0.0);

    // Step 2: Set number of threads
    omp_set_num_threads(numThreads);

    clock_t start = clock();

    // Step 3: Iteration loop
    for (int iter = 0; iter < maxIter; iter++) {

        // Step 4: Parallel Jacobi update
#pragma omp parallel for
        for (int i = 0; i < n; i++) {
            double sigma = 0.0;

            // Step 4.1: Sum A[i][j] * x[j]
            for (int j = 0; j < n; j++) {
                if (i != j) sigma += A[i][j] * x[j];
            }

            x_new[i] = (b[i] - sigma) / A[i][i];
        }

        // Step 5: Compute error (parallel reduction)
        double error = 0.0;
#pragma omp parallel for reduction(+:error)
        for (int i = 0; i < n; i++)
            error += fabs(x_new[i] - x[i]);

        // Step 6: Convergence check
        if (error < tol)
            break;

        // Step 7: Update x
        x = x_new;
    }

    clock_t end = clock();
    double sec = (double)(end - start) / CLOCKS_PER_SEC;

    // Step 8: Output time
    cout << "\n----------------------------------\n";
    cout << "OpenMP\n";
    cout << "Number of threads: " << numThreads << "\n";
    cout << "Time completed: " << sec << " s\n";
    cout << "----------------------------------\n";
}
