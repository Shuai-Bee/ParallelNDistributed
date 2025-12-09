#include <iostream>
#include <vector>
#include <cmath>
#include <time.h>
#include "jacobi_serial.h"

using namespace std;

void jacobi_serial(vector<vector<double>>& A, vector<double>& b,
    int n, int maxIter, double tol)
{
    // Step 1: Initialize guess vectors
    vector<double> x(n, 0.0), x_new(n, 0.0);

    clock_t start = clock();

    // Step 2: Iteration loop
    for (int iter = 0; iter < maxIter; iter++) {

        // Step 3: Jacobi update for all variables
        for (int i = 0; i < n; i++) {
            double sigma = 0.0;

            // Step 3.1: Sum A[i][j] * x[j]
            for (int j = 0; j < n; j++) {
                if (i != j) sigma += A[i][j] * x[j];
            }

            // Step 3.2: Compute new x value
            x_new[i] = (b[i] - sigma) / A[i][i];
        }

        // Step 4: Compute error
        double error = 0.0;
        for (int i = 0; i < n; i++)
            error += fabs(x_new[i] - x[i]);

        // Step 5: Convergence check
        if (error < tol)
            break;

        // Step 6: Update x
        x = x_new;
    }

    clock_t end = clock();
    double sec = (double)(end - start) / CLOCKS_PER_SEC;

    // Step 7: Output time
    cout << "\n----------------------------------\n";
    cout << "Serial\n";
    cout << "Time completed: " << sec << " s\n";
    cout << "----------------------------------\n";
}
