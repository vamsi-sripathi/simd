#!/bin/bash

set -x

rm -f *.out

icc -O3 -qopt-zmm-usage=high -fno-alias -std=c99 -qmkl -xCORE-AVX512 bench_max.c -DUSE_MAX_REF -o max_ref_icc.out
icc -O3 -qopt-zmm-usage=high -fno-alias -std=c99 -qmkl -xCORE-AVX512 bench_max.c -DUSE_MAX_IDX_TRACKING -o max_itrack.out
icc -O3 -qopt-zmm-usage=high -fno-alias -std=c99 -qmkl -xCORE-AVX512 bench_max.c -DUSE_MAX_BLK_TRACKING -o max_btrack.out

gcc -I${MKLROOT}/include -std=c99 -O3 -march=icelake-server -mprefer-vector-width=512 bench_max.c -DUSE_MAX_REF -o max_ref_gcc.out \
    -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl
clang -I${MKLROOT}/include -std=c99 -O3 -march=icelake-server -mprefer-vector-width=512 bench_max.c -DUSE_MAX_REF -o max_ref_clang.out \
    -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl
