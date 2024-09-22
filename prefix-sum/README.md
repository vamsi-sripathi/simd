## Prefix-Sum

- An optimized SIMD implementation for computing prefix-sum (also known as scan) using Intel AVX512 intrinsics
- Benchmark to compare the performance of custom written AVX512 implementation against Intel C Compiler (ICC), GCC, Clang and NumPy.

![prefixsum-op](https://user-images.githubusercontent.com/18724658/166134832-2a29068f-3e73-4519-9a8e-15c17344379f.png)

## AVX512 Prefix-Sum Instruction Sequence:

![prefixsum-avx512](https://user-images.githubusercontent.com/18724658/166134744-d29c1d98-880c-4d2c-b2f9-7b52500cf58f.png)


## Prerequisites:
Intel C Compiler and Intel Math Kernel Library

## Build: 
Execute `build.sh`. This would produce the several binaries for computing prefix-sum,
- avx512.out: Explicitly written AVX512 intrinsics kernel
- omp.out: Intel Compiler optimized implementation through OpenMP SIMD directives
- ref.out: Baseline implementation that solely relies on Intel Compiler optimizer
- gcc-ref.out: Baseline implementation that solely relies on GCC
- gcc-omp.out: GCC optimized implementation through OpenMP SIMD directives
- clang-ref.out: Baseline implementation that solely relies on Clang/LLVM
- clang-omp.out: Clang optimized implementation through OpenMP SIMD directives

In addition, it also produces a shared library (lib_par_avx512_psum.so) that provides a C interface to benchmark against NumPy prefix-sum/cumsum (cumulative sum)

## Run: 
All the binaries produced in the build step accept 3 command-line arguments specifying the start, end and step size of input vectors.
```
USAGE: ./avx512.out <start-size> <end-size> <step-size>
        {start, end, step}-size ->  Integers controlling the size of inputs used in the benchmark
```

## Benchmark:
- Running the script `bench.sh` would do a sweep from 64 to 1024 input sizes in steps of 16 (all in L1$) and dumps the stats to data file (plot.dat).
- NumPy: Set OMP_NUM_THREADS=1 before running bench_psum.py to benchmark against NumPy (since NumPy is sequential).
- Numba: See bench_psum_sdc.py

## Results: 
- The average speed-up of the explicit SIMD prefix-sum implementation over standard Compilers/runtime (GCC, Clang, Intel, NumPy) is ~5x on Intel Xeon Icelake server.
- GCC and Clang are unable to vectorize the prefix-sum computations. Their performance remains unchanged, even with the OpenMP SIMD directives.
- Intel C Compiler does a great job of auto-vectorization when OpenMP SIMD directives are used. Hence, the speed-up of explicit AVX512 implmentation is lower at 1.5x


Full technical article appeared in Parallel Universe Magazine Issue #44 [(link)](https://www.intel.com/content/www/us/en/developer/articles/technical/optimize-scan-operations-explicit-vectorization.html) [(pdf)](./parallel-universe-issue-44.pdf)
