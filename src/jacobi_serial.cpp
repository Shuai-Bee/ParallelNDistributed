#include <iostream>
#include <vector>
#include <cmath>
#include <time.h>
#include "jacobi_serial.h"

using namespace std;

// ---------------------------------------------------------------
// jacobi_serial()
// Performs the Jacobi iterative method without parallelization.
// A: coefficient matrix
// b: right-hand-side vector
// n: matrix size
// maxIter: maximum number of iterations
// tol: stopping tolerance
// ---------------------------------------------------------------
void jacobi_serial(vector<vector<double>>& A, vector<double>& b, int n, int maxIter, double tol)
{
    // x stores current solution guess
    // x_new stores updated solution values
    vector<double> x(n, 0.0), x_new(n, 0.0);

    clock_t start = clock();  // start timing

    // Main iteration loop
    for (int iter = 0; iter < maxIter; iter++) {

        // Update each variable using Jacobi formula
        for (int i = 0; i < n; i++) {
            double sigma = 0.0;

            // Compute sum of A[i][j] * x[j] for j not equal to i
            for (int j = 0; j < n; j++) {
                if (i != j) sigma += A[i][j] * x[j];
            }

            // Compute new x value
            x_new[i] = (b[i] - sigma) / A[i][i];
        }

        // Compute error = sum of absolute differences
        double error = 0.0;
        for (int i = 0; i < n; i++)
            error += fabs(x_new[i] - x[i]);

        // Check convergence
        if (error < tol)
            break;

        // Update solution vector
        x = x_new;
    }

    clock_t end = clock();  // stop timing
    double sec = (double)(end - start) / CLOCKS_PER_SEC;

    cout << "\n----------------------------------\n";
    cout << "Serial\n";
    cout << "----------------------------------\n";
	cout << "Matrix size: " << n << " x " << n << "\n";
    cout << "Time completed: " << sec << " s\n";
    cout << "----------------------------------\n";

}
