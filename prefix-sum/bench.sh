#!/bin/bash

set -x

rm -f *.log

strt=64
end=1024
step=16

taskset -c 4 numactl -m0 ./gcc-ref.out   $strt $end $step >> gcc-ref.log
taskset -c 4 numactl -m0 ./gcc-omp.out   $strt $end $step >> gcc-omp.log
taskset -c 4 numactl -m0 ./clang-ref.out $strt $end $step >> clang-ref.log
taskset -c 4 numactl -m0 ./clang-omp.out $strt $end $step >> clang-omp.log
taskset -c 4 numactl -m0 ./ref.out       $strt $end $step >> intel-ref.log
taskset -c 4 numactl -m0 ./omp.out       $strt $end $step >> intel-omp.log
taskset -c 4 numactl -m0 ./avx512.out    $strt $end $step >> avx512.log

grep -i fail *.log

awk '/perf/ {print $4}' gcc-ref.log | tr -d , > c1.dat

awk '/perf/ {print $(NF)}' gcc-ref.log   | tr -d , > c2.dat
awk '/perf/ {print $(NF)}' gcc-omp.log   | tr -d , > c3.dat
awk '/perf/ {print $(NF)}' clang-ref.log | tr -d , > c4.dat
awk '/perf/ {print $(NF)}' clang-omp.log | tr -d , > c5.dat
awk '/perf/ {print $(NF)}' intel-ref.log | tr -d , > c6.dat
awk '/perf/ {print $(NF)}' intel-omp.log | tr -d , > c7.dat
awk '/perf/ {print $(NF)}' avx512.log    | tr -d , > c8.dat

paste c1.dat c2.dat c3.dat c4.dat c5.dat c6.dat c7.dat c8.dat > tmp.dat

echo -e "#N\tGCC-ref\tGCC-OMP\tCLANG-ref\tCLANG-OMP\tICC-ref\tICC-OMP\tExplicit-AVX512" > plot.dat
awk '{printf("%d\t %d\t%d\t %d\t%d\t %d\t%d\t %d\t %.2f\t%.2f\t %.2f\t%.2f\t %.2f\t%.2f\n",$1,$2,$3,$4,$5,$6,$7,$8,$2/$8,$3/$8,$4/$8,$5/$8,$6/$8,$7/$8) }' tmp.dat >> plot.dat
 
awk 'BEGIN{s=0;} {s+=$9}  END{printf("Average speed-up over Baseline + GCC = %.2f\n", s/NR)}'        plot.dat
awk 'BEGIN{s=0;} {s+=$10} END{printf("Average speed-up over OMP SIMD Scan + GCC = %.2f\n", s/NR)}'   plot.dat
awk 'BEGIN{s=0;} {s+=$11} END{printf("Average speed-up over Baseline + Clang = %.2f\n", s/NR)}'      plot.dat
awk 'BEGIN{s=0;} {s+=$12} END{printf("Average speed-up over OMP SIMD Scan + Clang = %.2f\n", s/NR)}' plot.dat
awk 'BEGIN{s=0;} {s+=$13} END{printf("Average speed-up over Baseline + ICC = %.2f\n", s/NR)}'        plot.dat
awk 'BEGIN{s=0;} {s+=$14} END{printf("Average speed-up over OMP SIMD Scan + ICC = %.2f\n", s/NR)}'   plot.dat

rm c*.dat tmp.dat
