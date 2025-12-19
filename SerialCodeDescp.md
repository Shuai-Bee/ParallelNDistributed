## Jacobi Iterative Method

 Loop fraction in sequential in Jacobi Interaction Method
 
 To view the full version of code, visit here:
 https://github.com/Shuai-Bee/ParallelNDistributed/blob/main/src/jacobi_serial.cpp

```cpp
for (int iter = 0; iter < maxIter; iter++) {
    // Update each variable using Jacobi formula
    for (int i = 0; i < n; i++) {
        double sigma = 0.0;

        // Compute sum of A[i][j] * x[j] for j ≠ i
        for (int j = 0; j < n; j++) {
            if (i != j)
                sigma += A[i][j] * x[j];
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
