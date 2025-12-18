#include <iostream>
#include <vector>
#include <iomanip>
#include <string>
#include <omp.h>
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
    string input;
    int n, numThreads;
	int maxIter = 5000;
    double tol = 1e-4;

    printTitle();
    // User input: matrix size, threads
    while (true) {
        cout << "Enter matrix size (e.g. 100, 300, 500): ";
        cin >> input;
        
        if (isdigit(input[0])) {
            n = stoi(input);
            break;
        }
        else {
			cout << "Input must be digit only" << endl;
        }
	}

    while (true) {
        cout << "Enter number of threads (e.g. 1, 2, 4, 8): ";
        cin >> input;

        if (isdigit(input[0])) {
            numThreads = stoi(input);
            break;
        }
        else {
            cout << "Input must be digit only" << endl;
        }
    }

    // Create matrix A and vector b
    vector<vector<double>> A(n, vector<double>(n));
    vector<double> b(n);

    // Generate diagonally dominant matrix
    // This ensures Jacobi method converges
    for (int i = 0; i < n; i++) {
         double rowsum = 0.0;
         for (int j = 0; j < n; j++) {
             A[i][j] = rand() % 10 + 1;
             if (i != j) rowsum += A[i][j];
         }
         // Increase diagonal value so that A[i][i] > sum of other elements
         A[i][i] = rowsum + rand() % 5 + 5;
         b[i] = rand() % 20 + 5;
    }

    // Run serial version
    cout << "\nRunning SERIAL Jacobi..." << endl;
    jacobi_serial(A, b, n, maxIter, tol);

    // Run OpenMP parallel version
    cout << "\nRunning OPENMP Jacobi..." << endl;

        // Test different scheduling modes
    for (const string& schedName : schedules) {
        cout << "\n--- Mode: " << schedName << " ---" << endl;
        jacobi_omp(A, b, n, maxIter, tol, numThreads, schedName);
    }
	return 0;
}
