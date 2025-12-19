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
For Example:  
- 100 → small  
- 300 → medium  
- 500 → large

### 2. Tolerance
The defualt tolerance used in this code 1e-4 (0.0001), which have good accuracy also run faster 

### 3. Number of Threads
This value is used in the OpenMP version.  
It decides how many CPU cores will run at the same time.  
Examples:  
- 1 thread → no parallel speed-up  
- 2 threads → medium speed  
- 4 threads → faster (depends on CPU)
- User may enter the number of threads they wish to use 

---

## Program Output
After running, the program will show: 
- Execution time for serial version  
- Execution time for OpenMP version  
  
This allows us to study how parallel computing improves the Jacobi method.
