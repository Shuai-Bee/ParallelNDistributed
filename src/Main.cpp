#include <iostream>
#include <chrono>
#include "jacobi.h"

using namespace std;
using namespace std::chrono;

int main() {
    int n;
    int maxIter;
    double tol;
    int numThreads;

    // ----- User Input -----
    cout << "Enter matrix size (e.g. 100, 300, 500): ";
    cin >> n;

    cout << "Enter max iterations (e.g. 5000 or 10000): ";
    cin >> maxIter;

    cout << "Enter tolerance (e.g. 1e-4 or 1e-6): ";
    cin >> tol;

    cout << "Enter number of threads (e.g. 1, 2, 4, 8): ";
    cin >> numThreads;

    // Prepare system
    vector<vector<double>> A;
    vector<double> b;

    srand(0);
    generateSystem(n, A, b);

    cout << "\n==============================\n";
    cout << "Matrix size: " << n << endl;
    cout << "Max iterations: " << maxIter << endl;
    cout << "Tolerance: " << tol << endl;
    cout << "Threads: " << numThreads << endl;
    cout << "==============================\n";

    // Serial Jacobi
    vector<double> x_serial;
    auto startS = high_resolution_clock::now();
    int iterS = jacobiSerial(A, b, x_serial, maxIter, tol);
    auto endS = high_resolution_clock::now();
    auto timeS = duration_cast<milliseconds>(endS - startS).count();

    cout << "\nSerial Jacobi:" << endl;
    cout << "Iterations: " << iterS << endl;
    cout << "Time: " << timeS << " ms\n";

    // OpenMP Jacobi
    vector<double> x_omp;
    auto startO = high_resolution_clock::now();
    int iterO = jacobiOpenMP(A, b, x_omp, maxIter, tol, numThreads);
    auto endO = high_resolution_clock::now();
    auto timeO = duration_cast<milliseconds>(endO - startO).count();

    cout << "\nOpenMP Jacobi (" << numThreads << " threads):" << endl;
    cout << "Iterations: " << iterO << endl;
    cout << "Time: " << timeO << " ms\n";

    // Difference
    double err = computeError(x_serial, x_omp);
    cout << "\nDifference between Serial and OpenMP: " << err << endl;

    if (timeO > 0) {
        double speedup = (double)timeS / (double)timeO;
        double efficiency = speedup / numThreads;

        cout << "Speedup: " << speedup << endl;
        cout << "Efficiency: " << efficiency << endl;
    }
    else {
        cout << "OpenMP time is too small to calculate speedup and efficiency." << endl;
	}
    cout << "\nProgram Finished.\n";

    return 0;
}
