#include "jacobi.h"
#include <cmath>
#include <iostream>
#include <cstdlib>
#include <chrono>

#ifdef _OPENMP

#include <omp.h> // Include OpenMP header

#endif // _OPENMP

//Create a diagonally dominant matrix A and vector b for testing
void generateSystem(
    int n,
    vector<vector<double>>& A,
    vector<double>& b
) {
    A.assign(n, vector<double>(n, 0.0));
    b.assign(n, 0.0);

    vector<double> x_true(n, 1.0);

    for (int i = 0; i < n; ++i) {
        double row_sum = 0.0;
        for (int j = 0; j < n; ++j) {
            if (i == j) continue;
            double val = (rand() % 5) + 1;
            A[i][j] = val;
            row_sum += fabs(val);
        }
        A[i][i] = row_sum + (rand() % 5) + 5;
    }

    for (int i = 0; i < n; ++i) {
        double sum = 0.0;
        for (int j = 0; j < n; ++j) {
            sum += A[i][j] * x_true[j];
        }
        b[i] = sum;
    }
}

// Serial Jacobi Method Implementation
int jacobiSerial(
    const vector<vector<double>>& A,
    const vector<double>& b,
    vector<double>& x,
    int maxIter,
    double tol
) {
    int n = (int)A.size();
    vector<double> x_old(n, 0.0);
    x.assign(n, 0.0);

    for (int iter = 0; iter < maxIter; ++iter) {
        x_old = x;

        double maxDiff = 0.0;

        for (int i = 0; i < n; ++i) {
            double sum = 0.0;
            for (int j = 0; j < n; ++j) {
                if (j != i)
                    sum += A[i][j] * x_old[j];
            }

            x[i] = (b[i] - sum) / A[i][i];

            double diff = fabs(x[i] - x_old[i]);
            if (diff > maxDiff)
                maxDiff = diff;
        }

        if (maxDiff < tol)
            return iter + 1;
    }

    return maxIter;
}

// Jacobi Method Implementation with OpenMP
int jacobiOpenMP(
    const vector<vector<double>>& A,
    const vector<double>& b,
    vector<double>& x,
    int maxIter,
    double tol,
    int numThreads
) {
    int n = (int)A.size();
    vector<double> x_old(n, 0.0);
    x.assign(n, 0.0);

#ifdef _OPENMP
    omp_set_num_threads(numThreads);
#endif

    for (int iter = 0; iter < maxIter; ++iter) {
        x_old = x;

        double maxDiff = 0.0;

        #pragma omp parallel
        {
            double localMaxDiff = 0.0;

            #pragma omp for
            for (int i = 0; i < n; ++i) {
                double sum = 0.0;
                for (int j = 0; j < n; ++j) {
                    if (j != i)
                        sum += A[i][j] * x_old[j];
                }

                double newVal = (b[i] - sum) / A[i][i];
                double diff = fabs(newVal - x_old[i]);

                x[i] = newVal;

                if (diff > localMaxDiff)
                    localMaxDiff = diff;
            }

#pragma omp critical
            {
                if (localMaxDiff > maxDiff)
                    maxDiff = localMaxDiff;
            }
        }

        if (maxDiff < tol)
            return iter + 1;
    }

    return maxIter;
}

double computeError(const vector<double>& x1,
    const vector<double>& x2) {
    double maxDiff = 0.0;

    for (size_t i = 0; i < x1.size(); ++i) {
        double diff = fabs(x1[i] - x2[i]);
        if (diff > maxDiff)
            maxDiff = diff;
    }
    return maxDiff;
}