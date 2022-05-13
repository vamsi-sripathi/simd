#!/bin/bash

set -x

rm -f *.o *.out *.so

icc -c  -O3 -xCORE-AVX512 common.c -o common.o

icc -c  -O3 -qmkl -xCORE-AVX512 -DUSE_REF bench_psum.c -o bench_psum_ref.o
icc -c  -O3 -xCORE-AVX512 -qopt-zmm-usage=high -DUSE_REF kernels.c -o kernels_ref.o
icc -qmkl -O3 -xCORE-AVX512 bench_psum_ref.o kernels_ref.o common.o -o ref.out

icc -c  -O3 -qmkl -xCORE-AVX512 -DUSE_OMP bench_psum.c -o bench_psum_omp.o 
icc -c  -O3 -xCORE-AVX512 -qopt-zmm-usage=high -DUSE_OMP kernels.c -o kernels_omp.o 
icc -qmkl -O3 -xCORE-AVX512 bench_psum_omp.o kernels_omp.o common.o -o omp.out

icc -c  -O3 -qmkl -xCORE-AVX512 -DUSE_AVX512 bench_psum.c -o bench_psum_avx512.o 
icc -c  -O3 -xCORE-AVX512 -qopt-zmm-usage=high -DUSE_AVX512 kernels.c -o kernels_avx512.o 
icc -qmkl -O3 -xCORE-AVX512 bench_psum_avx512.o kernels_avx512.o common.o -o avx512.out

# icc -c  -O3 -xCORE-AVX512 -qopt-zmm-usage=high -DUSE_AVX512 -DOPT2 kernels.c -o kernels_avx512.o 
# icc -qmkl -O3 -xCORE-AVX512 bench_psum_avx512.o kernels_avx512.o common.o -o avx512.opt2.out

icc -qopenmp -fPIC -fp-model strict -c -std=c99 -O3 -xCORE-AVX512 -qopt-zmm-usage=high -DUSE_AVX512 -DUSE_THREADS kernels.c -o par_avx512_psum.o
icc -shared -fPIC -O3 -xCORE-AVX512 par_avx512_psum.o -o lib_par_avx512_psum.so

for CC in gcc clang
do
  $CC -c  -O3 common.c -o common_${CC}.o
  $CC -I${MKLROOT}/include -c  -O3 -march=skylake-avx512 -DUSE_REF bench_psum.c -o bench_psum_ref_${CC}.o
  $CC -c  -O3 -march=skylake-avx512 -DUSE_REF kernels.c -o kernels_ref_${CC}.o
  $CC -O3 -march=skylake-avx512 bench_psum_ref_${CC}.o kernels_ref_${CC}.o common_${CC}.o -o ${CC}-ref.out -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl

  $CC -I${MKLROOT}/include -c  -O3 -march=skylake-avx512 -DUSE_OMP bench_psum.c -o bench_psum_omp_${CC}.o
  $CC -c  -O3 -march=skylake-avx512 -DUSE_OMP kernels.c -o kernels_omp_${CC}.o
  $CC -O3 -march=skylake-avx512 bench_psum_omp_${CC}.o kernels_omp_${CC}.o common_${CC}.o -o ${CC}-omp.out -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl
done
