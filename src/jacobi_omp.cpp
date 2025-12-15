#include <iostream>
#include <vector>
#include <cmath>
#include <time.h>
#include <omp.h>
#include "jacobi_omp.h"


// add base line for iteration
using namespace std;

// ---------------------------------------------------------------
// jacobi_omp()
// Performs the Jacobi method using OpenMP for parallel execution.
// A: matrix
// b: constant vector
// numThreads: number of threads to use
// ---------------------------------------------------------------
void jacobi_omp(vector<vector<double>>& A, vector<double>& b, int n, int maxIter, double tol, int numThreads)
{
    vector<double> x(n, 0.0), x_new(n, 0.0);

    // Set OpenMP thread count
    omp_set_num_threads(numThreads);

    clock_t start = clock();  // start timing

    // Main iteration loop
    for (int iter = 0; iter < maxIter; iter++) {

        // Parallel Jacobi update loop
    #pragma omp parallel for
        for (int i = 0; i < n; i++) {
			double sigma = 0.0;             

            // Compute sum of other terms in the row
            for (int j = 0; j < n; j++) {
                if (i != j) sigma += A[i][j] * x[j];
            }

            x_new[i] = (b[i] - sigma) / A[i][i];
        }

        // Parallel reduction to compute total error
        double error = 0.0;
    #pragma omp parallel for reduction(+:error)
        for (int i = 0; i < n; i++)
            error += fabs(x_new[i] - x[i]);

        // Stop if converged
        if (error < tol)
            break;

        // Update solution vector
        x = x_new;
    }

    clock_t end = clock();  // stop timing
    double sec = (double)(end - start) / CLOCKS_PER_SEC;

    cout << "\n----------------------------------\n";
    cout << "OpenMP\n";
    cout << "----------------------------------\n";
	cout << "Matrix size: " << n << " x " << n << "\n";
    cout << "Number of threads: " << numThreads << "\n";
    cout << "Time completed: " << sec << " s\n";
    cout << "----------------------------------\n";
   
}
