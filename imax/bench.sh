#!/bin/bash

set -x

rm -f *.log

# 4 KiB to 48 KiB
for ((n=1024;n<=$((1024*12));n+=1024));
do
  ./max_ref_gcc.out   $n 0 0      2>&1 | tee -a ref_gcc.log
  ./max_ref_clang.out $n 0 0      2>&1 | tee -a ref_clang.log
  ./max_ref_icx.out   $n 0 0      2>&1 | tee -a ref_icx.log
  ./max_btrack.out    $n 0 1024   2>&1 | tee -a btrack.log
  ./max_itrack.out    $n 0 0      2>&1 | tee -a itrack.log
  ./max_mkl.out       $n 0 0      2>&1 | tee -a mkl.log
done

# 64 KiB to 1 MiB
echo "==="
for ((n=16384;n<$((16384*16));n+=16384));
do
  ./max_ref_gcc.out   $n 0 0      2>&1 | tee -a ref_gcc.log
  ./max_ref_clang.out $n 0 0      2>&1 | tee -a ref_clang.log
  ./max_ref_icx.out   $n 0 0      2>&1 | tee -a ref_icx.log
  ./max_btrack.out    $n 0 1024   2>&1 | tee -a btrack.log
  ./max_itrack.out    $n 0 0      2>&1 | tee -a itrack.log
  ./max_mkl.out       $n 0 0      2>&1 | tee -a mkl.log
done

# 1 MiB to 16 MiB
echo "==="
st=$((16384*16))
for ((n=st;n<=$((st*16));n+=st));
do
  ./max_ref_gcc.out   $n 0 0      2>&1 | tee -a ref_gcc.log
  ./max_ref_clang.out $n 0 0      2>&1 | tee -a ref_clang.log
  ./max_ref_icx.out   $n 0 0      2>&1 | tee -a ref_icx.log
  ./max_btrack.out    $n 0 1024   2>&1 | tee -a btrack.log
  ./max_itrack.out    $n 0 0      2>&1 | tee -a itrack.log
  ./max_mkl.out       $n 0 0      2>&1 | tee -a mkl.log
done;

awk '/Perf/ {print $4}'      ref_gcc.log   | tr -d "," > n.dat
awk '/Perf/ {print $(NF-3)}' ref_gcc.log   | tr -d "," > t_gcc.dat
awk '/Perf/ {print $(NF-3)}' ref_clang.log | tr -d "," > t_clang.dat
awk '/Perf/ {print $(NF-3)}' ref_icx.log   | tr -d "," > t_icx.dat
awk '/Perf/ {print $(NF-3)}' btrack.log    | tr -d "," > t_btrack.dat
awk '/Perf/ {print $(NF-3)}' itrack.log    | tr -d "," > t_itrack.dat
awk '/Perf/ {print $(NF-3)}' mkl.log       | tr -d "," > t_mkl.dat

echo -e "N\tGCC\tClang\tICX\tAVX512_2PASS\tAVX512_1PASS\tMKL" > results.tsv
paste n.dat t_gcc.dat t_clang.dat t_icx.dat t_btrack.dat t_itrack.dat t_mkl.dat >> results.tsv

rm -rf *.dat
