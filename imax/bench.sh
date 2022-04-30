#!/bin/bash

set -x

rm -f *.log

for ((n=1024;n<=$((1024*12));n+=1024));
do
  ./max_ref_gcc.out $n 0 0 2>&1 | tee -a ref_gcc.log
  ./max_ref_clang.out $n 0 0 2>&1 | tee -a ref_clang.log
  ./max_ref_icc.out $n 0 0 2>&1 | tee -a ref_icc.log
  # ./max_btrack.out $n 0 1024 2>&1 | tee -a btrack.log
  ./max_itrack.out $n 0 0 2>&1 | tee -a itrack.log
done

echo "==="
for ((n=16384;n<$((16384*16));n+=16384));
do
  ./max_ref_gcc.out $n 0 0 2>&1 | tee -a ref_gcc.log
  ./max_ref_clang.out $n 0 0 2>&1 | tee -a ref_clang.log
  ./max_ref_icc.out $n 0 0 2>&1 | tee -a ref_icc.log
  # ./max_btrack.out $n 0 1024 2>&1 | tee -a btrack.log
  ./max_itrack.out $n 0 0 2>&1 | tee -a itrack.log
done

echo "==="
st=$((16384*16))
for ((n=st;n<=$((st*16));n+=st));
do
  ./max_ref_gcc.out $n 0 0 2>&1 | tee -a ref_gcc.log
  ./max_ref_clang.out $n 0 0 2>&1 | tee -a ref_clang.log
  ./max_ref_icc.out $n 0 0 2>&1 | tee -a ref_icc.log
  # ./max_btrack.out $n 0 1024 2>&1 | tee -a btrack.log
  ./max_itrack.out $n 0 0 2>&1 | tee -a itrack.log
done;

awk '/Perf/ {print $4}' ref_gcc.log | tr -d "," > t0
awk '/Perf/ {print $(NF-3)}' ref_gcc.log | tr -d "," > t1
awk '/Perf/ {print $(NF-3)}' ref_clang.log | tr -d ","  > t2
awk '/Perf/ {print $(NF-3)}' ref_icc.log | tr -d ","  > t3
# awk '/Perf/ {print $NF}' btrack.log > t4
awk '/Perf/ {print $(NF-3)}' itrack.log | tr -d ","  > t5

echo -e "N\tGCC\tCLANG\tAVX512" > results.tsv
paste t0 t1 t2 t5 >> results.tsv
rm -rf t[0-5]
