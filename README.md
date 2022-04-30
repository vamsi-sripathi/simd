# simd
This repository contains a collection of optimized SIMD implementations for popular patterns that I came across in my various projects.

- imax: Find the index of the maximum element in a dataset.
- Prefix-sum: Calculate the cumulative sum


## IMAX:
The imax directory contains a optimized AVX512 SIMD implementation and a benchmark to compare the performance against GCC and Clang. The average speed-up over GCC/Clang for sizes that fit in L1, L2, and L3 caches are 18x, 40x, and 13x, respectively. Full technical article appeared in Parallel Universe Magazine Issue #46 [(link)](https://www.intel.com/content/www/us/en/developer/community/parallel-universe-magazine/overview.html) [(pdf)](./imax/parallel-universe-issue-46.pdf)


![Optimized maxloc](https://user-images.githubusercontent.com/18724658/166125947-7ac722d1-852d-49c7-a1d4-54ce14f03d49.png)
