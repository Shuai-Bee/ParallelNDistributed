#ifndef JACOBI_SERIAL_H
#define JACOBI_SERIAL_H

#include <vector>
using namespace std;

// Serial Jacobi method.
// Solves Ax = b using the Jacobi iterative formula.
void jacobi_serial(vector<vector<double>>& A,
    vector<double>& b,
    int n, int maxIter, double tol);

#endif
