## IMAX:

- An optimized AVX512 SIMD implementation for finding the index of the maximum element in a dataset.
- Written using Intel AVX512 intrinsics
- Two implementations:
  - Single Pass: This methods implictly tracks the target index location and does least number of compare operations.
  - Two Pass: One pass to find the block containing the maximum value. Revisit the block again to find the index corresponding to maximum value.
- Benchmark to compare the performance against GCC and Clang.

Prerequisites:
Intel C Compiler and Intel Math Kernel Library

Build: Run `build.sh`

Run: 



The average speed-up over GCC/Clang for sizes that fit in L1, L2, and L3 caches are 18x, 40x, and 13x, respectively. Full technical article appeared in Parallel Universe Magazine Issue #46 [(link)](https://www.intel.com/content/www/us/en/developer/community/parallel-universe-magazine/overview.html) [(pdf)](./imax/parallel-universe-issue-46.pdf)


![Optimized maxloc](https://user-images.githubusercontent.com/18724658/166125947-7ac722d1-852d-49c7-a1d4-54ce14f03d49.png)
