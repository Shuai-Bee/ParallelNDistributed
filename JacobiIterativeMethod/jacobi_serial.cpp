#include <iostream>
#include <vector>
#include <cmath>
#include <time.h>
#include "jacobi_serial.h"

using namespace std;

void jacobi_serial(vector<vector<double>>& A, vector<double>& b,
    int n, int maxIter)
{
    // Step 0: Set default tolerance
    double tol = 1e-4;

    // Step 1: Initialize guess vectors
    vector<double> x(n, 0.0), x_new(n, 0.0);

    // Step 2: Start timer
    clock_t start = clock();

    // Step 3: Iteration loop
    for (int iter = 0; iter < maxIter; iter++) {

        // Step 3.1: Jacobi update for each variable
        for (int i = 0; i < n; i++) {
            double sigma = 0.0;
            for (int j = 0; j < n; j++)
                if (i != j) sigma += A[i][j] * x[j];
            x_new[i] = (b[i] - sigma) / A[i][i];
        }

        // Step 3.2: Compute error
        double error = 0.0;
        for (int i = 0; i < n; i++)
            error += fabs(x_new[i] - x[i]);

        // Step 3.3: Convergence check
        if (error < tol)
            break;

        // Step 3.4: Update solution
        x = x_new;
    }

    // Step 4: Stop timer
    clock_t end = clock();
    double sec = (double)(end - start) / CLOCKS_PER_SEC;

    // Step 5: Output results
    cout << "\n----------------------------------\n";
    cout << "Serial Jacobi Method\n";
    cout << "Matrix size: " << n << " x " << n << "\n";
    cout << "Max iterations: " << maxIter << "\n";
    cout << "Tolerance: " << tol << "\n";
    cout << "Time completed: " << sec << " s\n";
    cout << "----------------------------------\n";
}
