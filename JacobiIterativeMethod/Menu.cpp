#include <iostream>
#include <vector>
#include <iomanip>
#include <cstdlib>   
#include "jacobi_serial.h"
#include "jacobi_omp.h"

using namespace std;

void printTitle() {
    string title = "Jacobi Iterative Method";
    int width = 46; // total width of the border line

    // Print top border
    cout << string(width, '+') << endl;

    // Print centered title
    cout << setw((width + title.length()) / 2) << title << endl;

    // Print bottom border
    cout << string(width, '+') << endl;
}

int main() {
    int n, maxIter, numThreads;

    printTitle();

    // User input: matrix size, max iterations, threads
    cout << "Enter matrix size (e.g. 100, 300, 500): ";
    cin >> n;

    cout << "Enter max iterations (e.g. 5000 or 10000): ";
    cin >> maxIter;

    cout << "Enter number of threads (e.g. 1, 2, 4, 8): ";
    cin >> numThreads;

    // Create matrix A and vector b
    vector<vector<double>> A(n, vector<double>(n));
    vector<double> b(n);

    // Generate diagonally dominant matrix
    for (int i = 0; i < n; i++) {
        double rowsum = 0.0;
        for (int j = 0; j < n; j++) {
            A[i][j] = rand() % 10 + 1;
            if (i != j) rowsum += A[i][j];
        }
        // Make diagonal dominant
        A[i][i] = rowsum + rand() % 5 + 5;
        b[i] = rand() % 20 + 5;
    }

    // Run serial version
    cout << "\nRunning SERIAL Jacobi..." << endl;
    jacobi_serial(A, b, n, maxIter);

    // Run OpenMP parallel version
    cout << "\nRunning OPENMP Jacobi..." << endl;
    jacobi_omp(A, b, n, maxIter, numThreads);

    return 0;
}
