## Prefix-Sum
![prefixsum-op](https://user-images.githubusercontent.com/18724658/166134832-2a29068f-3e73-4519-9a8e-15c17344379f.png)

This benchmark compares the performance of custom written prefix-sum kernel that explicitly uses AVX512 instructions against existing implementations from different Compilers (Intel C Compiler, GCC, Clang). Below figure shows the AVX512 instruction sequence.

![prefixsum-avx512](https://user-images.githubusercontent.com/18724658/166134744-d29c1d98-880c-4d2c-b2f9-7b52500cf58f.png)


## Prerequisites:
Intel C Compiler and Intel Math Kernel Library

## Build: 
Execute `build.sh`. This would produce the several binaries for computing prefix-sum,
- avx512.out: Explicitly written AVX512 intrinsics kernel
- omp.out: Compiler optimized implementation through OpenMP SIMD directives
- ref.out: Baseline implementation that solely relies on Compiler optimizer

## Run: 
All the binaries produced in the build step accept 3 command-line arguments specifying the start, end and step size of input vectors.
```
USAGE: ./avx512.out <start-size> <end-size> <step-size>
        {start, end, step}-size ->  Integers controlling the size of inputs used in the benchmark
```

## Benchmark:
Running the script `bench.sh` would do a sweep from 64 to 1024 input sizes in steps of 16 (all in L1$) and dumps the stats to data file (plot.dat).


## Results: 
- The explicit AVX-512 SIMD implementation outperforms both the baseline and OpenMP SIMD implementations from Intel Compiler, GCC and Clang.
- GCC and Clang are unable to vectorize the prefix-sum computations. Their performance remains unchanged, even with the OpenMP SIMD directives.
- Intel C Compiler does a great job of auto-vectorization when OpenMP SIMD directives are used.
- The average speed-up of the explicit SIMD prefix-sum implementation over the baseline and OpenMP SIMD is 4.6x (GCC and Clang) and 1.6x (Intel C++ Compiler) respectively on Intel Xeon (Icelake) server.


Full technical article appeared in Parallel Universe Magazine Issue #44 [(link)](https://www.intel.com/content/www/us/en/developer/community/parallel-universe-magazine/overview.html) [(pdf)](./parallel-universe-issue-44.pdf)
