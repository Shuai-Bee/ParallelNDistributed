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
void jacobi_omp(vector<vector<double>>& A, vector<double>& b, int n, int maxIter, double tol, int numThreads,string typeName)
{
    vector<double> x(n, 0.0), x_new(n, 0.0);

    // Set OpenMP thread count
    omp_set_num_threads(numThreads);

	double start = omp_get_wtime(); // start timing

    // Main iteration loop
	// OMP parallel for with different scheduling types
    for (int iter = 0; iter < maxIter; iter++) {
        if (typeName == "Static") {
        #pragma omp parallel for schedule(static)
            for (int i = 0; i < n; i++) {
                double sigma = 0.0;
                for (int j = 0; j < n; j++) {
                    if (i != j) sigma += A[i][j] * x[j];
                }
                x_new[i] = (b[i] - sigma) / A[i][i];
            }
        }
        else if (typeName == "Dynamic") {
        #pragma omp parallel for schedule(dynamic)
            for (int i = 0; i < n; i++) {
                double sigma = 0.0;
                for (int j = 0; j < n; j++) {
                    if (i != j) sigma += A[i][j] * x[j];
                }
                x_new[i] = (b[i] - sigma) / A[i][i];
            }
        }
        else if (typeName == "Guided") {
        #pragma omp parallel for schedule(guided)
            for (int i = 0; i < n; i++) {
                double sigma = 0.0;
                for (int j = 0; j < n; j++) {
                    if (i != j) sigma += A[i][j] * x[j];
                }
                x_new[i] = (b[i] - sigma) / A[i][i];
            }
        }
        else {
        #pragma omp parallel for schedule(runtime)
            for (int i = 0; i < n; i++) {
                double sigma = 0.0;
                for (int j = 0; j < n; j++) {
                    if (i != j) sigma += A[i][j] * x[j];
                }
                x_new[i] = (b[i] - sigma) / A[i][i];
            }
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

    double end = omp_get_wtime();  // stop timing
    double sec = (end - start);

    cout << "\n----------------------------------\n";
    cout << "OpenMP\n";
    cout << "----------------------------------\n";
	cout << "Matrix size: " << n << " x " << n << "\n";
    cout << "Number of threads: " << numThreads << "\n";
    cout << "OpenMP type: " << typeName << "\n";
    cout << "Time completed: " << sec << " s\n";
    cout << "----------------------------------\n";
   
}
