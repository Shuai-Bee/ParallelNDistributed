## Parallel Jacobi Iterative Method (OpenMP)

Loop fraction in OpenMP in Jacobi Interaction Method

```cpp
for (int iter = 0; iter < maxIter; iter++) {

    // Parallel Jacobi update loop
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        double sigma = 0.0;

        // Compute sum of other terms in the row
        for (int j = 0; j < n; j++) {
            if (i != j)
                sigma += A[i][j] * x[j];
        }

        // Compute new x value
        x_new[i] = (b[i] - sigma) / A[i][i];
    }

    // Parallel reduction to compute total error
    double error = 0.0;
    #pragma omp parallel for reduction(+:error)
    for (int i = 0; i < n; i++)
        error += fabs(x_new[i] - x[i]);

    // Stop if converged
    if (error < tol)
        break;

    // Update solution vector
    x = x_new;
}
