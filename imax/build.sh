#!/bin/bash

set -x
set -e

rm -f *.out

icx -O3 -fno-alias -std=c99 -qmkl -xCORE-AVX512 -qopt-zmm-usage=high bench_max.c -DUSE_MAX_REF -o max_ref_icx.out
icx -O3 -fno-alias -std=c99 -qmkl -xCORE-AVX512 -qopt-zmm-usage=high bench_max.c -DUSE_MAX_IDX_TRACKING -o max_itrack.out
icx -O3 -fno-alias -std=c99 -qmkl -xCORE-AVX512 -qopt-zmm-usage=high bench_max.c -DUSE_MAX_BLK_TRACKING -o max_btrack.out
icx -O3 -fno-alias -std=c99 -qmkl -xCORE-AVX512 -qopt-zmm-usage=high bench_max.c -DUSE_MAX_MKL -o max_mkl.out

gcc -O3 -std=c99 -march=icelake-server -mprefer-vector-width=512 bench_max.c -DUSE_MAX_REF -I${MKLROOT}/include -o max_ref_gcc.out \
    -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl
clang -O3 -std=c99 -march=icelake-server -mprefer-vector-width=512 bench_max.c -DUSE_MAX_REF -I${MKLROOT}/include -o max_ref_clang.out \
    -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl
