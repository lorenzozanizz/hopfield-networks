### Eigen Multithreading Benchmark Example

This example benchmarks Eigen's matrix multiplication to verify if multithreading is enabled and effective.

Key steps include:
- Defining a benchmark function that sets the number of threads, initializes Eigen parallel, creates two 2000x2000 random matrices, times the noalias multiplication, and returns the duration.
- In main, printing SIMD instruction sets in use.
- Running the benchmark with 1 thread and 8 threads, printing execution times.
- Checking if the 8-thread time is less than 70% of the 1-thread time to determine if multithreading is active.

The example integrates components from `math/`, `io/`, and `utils/` directories (though plotting is included but not used), providing a simple test to ensure we can have parallel performance in linear algebra operations.
This test may fail if the default BLAS library on the executing hardware is not multithreaded.