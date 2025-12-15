#include <iostream>
#include <vector>
#include <cmath>
#include <time.h>
#include <omp.h>
#include "jacobi_omp.h"

using namespace std;

void jacobi_omp(vector<vector<double>>& A, vector<double>& b,
    int n, int maxIter, int numThreads)
{
    // default tolerance
    double tol = 1e-4;

    // Step 1: Initialize guess vectors
    vector<double> x(n, 0.0), x_new(n, 0.0);

    // Step 2: Set number of threads
    omp_set_num_threads(numThreads);

    // Step 3: Start timer
    clock_t start = clock();

    // Step 4: Iteration loop
    for (int iter = 0; iter < maxIter; iter++) {

        // Step 4.1: Parallel Jacobi update
#pragma omp parallel for
        for (int i = 0; i < n; i++) {
            double sigma = 0.0;
            for (int j = 0; j < n; j++)
                if (i != j) sigma += A[i][j] * x[j];
            x_new[i] = (b[i] - sigma) / A[i][i];
        }

        // Step 4.2: Compute error (parallel reduction)
        double error = 0.0;
#pragma omp parallel for reduction(+:error)
        for (int i = 0; i < n; i++)
            error += fabs(x_new[i] - x[i]);

        // Step 4.3: Convergence check
        if (error < tol)
            break;

        // Step 4.4: Update solution
        x = x_new;
    }

    // Step 5: Stop timer
    clock_t end = clock();
    double sec = (double)(end - start) / CLOCKS_PER_SEC;

    // Step 6: Output results
    cout << "\n----------------------------------\n";
    cout << "OpenMP Jacobi Method\n";
    cout << "Matrix size: " << n << " x " << n << "\n";
    cout << "Number of threads: " << numThreads << "\n";
    cout << "Max iterations: " << maxIter << "\n";
    cout << "Tolerance: " << tol << "\n";
    cout << "Time completed: " << sec << " s\n";
    cout << "----------------------------------\n";
}
