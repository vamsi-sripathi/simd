## IMAX:

- An optimized SIMD implementation for finding the index of the maximum element in a dataset written using Intel AVX512 intrinsics
- Two implementations:
  - Single Pass: This method implictly tracks the target index location and does least number of compare operations.
  - Two Pass: One pass to find the block containing the maximum value. Revisit the block again to find the index corresponding to maximum value.
- Benchmark to compare the performance of custom written AVX512 implementation against Intel C Compiler (ICC), GCC and Clang.


![Optimized maxloc](https://user-images.githubusercontent.com/18724658/166125947-7ac722d1-852d-49c7-a1d4-54ce14f03d49.png)
Figure showing the AVX512 single pass implementation

## Prerequisites:
Intel C Compiler and Intel Math Kernel Library

## Build: 
- Execute `build.sh`
- This would produce the several binaries,
  - max_itrack: single-pass implementation
  - max_btrack: two-pass/blocking based implementation
  - max_ref_{icc, gcc, clang}: implementations relying on corresponding Compiler 

## Run: 
`
usage : ./max_itrack.out <size, int>   <data-order, 0|1|int>   <block-size, int>

        size -> Integer specifying the number of elements in input vector
        data-order -> Integer specifying on how-to initialize the input vector
                0 -> vector is initialized with values in ascending order
                1 -> vector is initialized with values in descending order
                any other integer -> the specified integer is used as random seed to fill entries
        block-size -> Used only in AVX512 blocking implementation
`

## Benchmark:
`bench.sh` is a script that launches the binaries for different sizes of input vectors to sweep the cache hierarchy.

## Results: 
- The average speed-up of AVX512 single-pass implementation over GCC/Clang for sizes that fit in L1, L2, and L3 caches are 18x, 40x, and 13x, respectively. 
- Intel Xeon Platinum 8368 processor (3rd Gen Intel Xeon Scalable processor) is used for benchmarking. It has 38 cores per socket, 48 KB L1D cache, 1280 KB L2, and 57 MB L3 per socket.
- GCC (8.4.1) and Clang (11.0.0) are used with the -O3 -march=icelake-server -mprefer-vector-width=512 flags to compile the baseline implementation. ICC (v19.1.3.304) is used to compile the AVX-512 implementation.
- The same input data is used in all benchmarks. The input data consists of values in ascending order (i.e., the maximum value is in the last index location). No performance differences were observed when random values were used.
- The input size is varied from 1,024 to 4,194,304 elements (16 MB with FP32 elements). This allows us to evaluate performance when the data fits into different cache hierarchies (L1D, L2, and L3).
- Performance is measured for one thread pinned to a single core. The average time elapsed for 100 iterations of each problem size is reported. Memory is allocated on 64B aligned boundaries.


Full technical article appeared in Parallel Universe Magazine Issue #46 [(link)](https://www.intel.com/content/www/us/en/developer/community/parallel-universe-magazine/overview.html) [(pdf)](./imax/parallel-universe-issue-46.pdf)
