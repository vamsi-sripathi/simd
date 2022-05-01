## Prefix-Sum

This benchmark compares the performance of custom written prefix-sum kernel that explicitly uses AVX512 instructions against existing implementations from Compilers (Intel C Compiler, GCC, Clang).

The AVX512 intrinsics kernel uses 11 adds and 10 permutes for computing prefix-sum on 16 elements. Some of the permutes are independent and are within a 128-bit lane. That leads to faster execution since vpermild has 1-cycle latency over vpermpd which has 3-cycle latency. In addition, an extra add is done to compute the accumulation for future iterations and reduce pipeline stalls.



## Prerequisites:
Intel C Compiler and Intel Math Kernel Library

The compilation step would produce 2 binaries (omp.out, awe.out). Both the binaries take 3 parameters specifying the start, end and step size of input vectors as commandline parameters -- ./{omp,awe}.out start end step

    omp.out --> Measure the time taken by OMP SIMD Scan for computing prefix-sum
    awe.out --> Measure the time taken by explicitly written AVX512 intrinsics kernelfor computing prefix-sum


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
Running the script `bench.sh` would do a sweep from 64 to 1024 in steps of 16 (all in L1$) and dumps the stats to data file (plot.dat).


## Results: 

Full technical article appeared in Parallel Universe Magazine Issue #44 [(link)](https://www.intel.com/content/www/us/en/developer/community/parallel-universe-magazine/overview.html) [(pdf)](./prefix-sum/parallel-universe-issue-44.pdf)
