#ifndef JACOBI_OMP_H
#define JACOBI_OMP_H

#include <vector>
using namespace std;

// Parallel Jacobi method using OpenMP threads.
void jacobi_omp(vector<vector<double>>& A,
    vector<double>& b,
    int n, int maxIter, double tol, int numThreads);

#endif
