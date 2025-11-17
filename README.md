# Parallel and Distributed Assignment

This project uses C++ to implement the Jacobi Iterative Method.  
The main purpose is to compare the performance of two versions:  
1. Serial version (single CPU core)  
2. OpenMP version (multi-core CPU)

The program will ask the user to enter several inputs before running the test.  
These inputs help control the accuracy and performance of the Jacobi method.

## User Inputs Explanation

### 1. Matrix Size (n)
This is the size of the square matrix A (n × n).  
A larger matrix means more work and longer execution time.  
Example:  
- 100 → small  
- 300 → medium  
- 500 → large

### 2. Maximum Iterations
This is the maximum number of times the algorithm will repeat.  
If the solution does not converge early, it will stop after this limit.

### 3. Tolerance
Tolerance is the stopping condition.  
It tells the program how close the new solution must be to the old solution.  
For example:  
- 1e-4 = 0.0001 (good accuracy, faster)  
- 1e-6 = 0.000001 (very high accuracy, slower)

If the change between iterations becomes smaller than the tolerance value, the algorithm stops.

### 4. Number of Threads
This value is used in the OpenMP version.  
It decides how many CPU cores will run at the same time.  
Examples:  
- 1 thread → no parallel speed-up  
- 2 threads → medium speed  
- 4 threads → faster (depends on CPU)

---

## Program Output
After running, the program will show:
- Number of iterations used  
- Execution time for serial version  
- Execution time for OpenMP version  
- Difference between the two results  
- Speedup  
- Efficiency

This allows us to study how parallel computing improves the Jacobi method.
