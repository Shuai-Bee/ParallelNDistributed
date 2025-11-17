#ifndef JACOBI_H
#define JACOBI_H

#include <vector>
using namespace std;

// Generate a diagonally dominant system Ax = b
void generateSystem(
    int n,
    vector<vector<double>>& A,
    vector<double>& b
);

// Serial Jacobi method
int jacobiSerial(
    const vector<vector<double>>& A,
    const vector<double>& b,
    vector<double>& x,
    int maxIter,
    double tol
);

// OpenMP Jacobi method
int jacobiOpenMP(const vector<vector<double>>& A,
    const vector<double>& b,
    vector<double>& x,
    int maxIter,
    double tol,
    int numThreads
);

// Compute difference between two solution vectors
double computeError(const vector<double>& x1,
    const vector<double>& x2
);

#endif