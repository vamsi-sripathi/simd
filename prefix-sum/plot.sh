#!/bin/bash

out_file=bench_psum
gnuplot <<- EOF
set terminal pngcairo enhanced
set grid
set key center
set xtics 32 rotate
#set xrange [64:1024]
#set yrange [0:]
set output "./${out_file}.png"
set title "Performance of Explicit AVX512 SIMD Prefix\-Sum/Scan"
set xlabel "Input Vector Size"
set ylabel "Speed-up"
plot "./plot.dat" using 1:9  with linespoints lw 3 title "Speed-up over Baseline + GCC",\
     "./plot.dat" using 1:10 with linespoints lw 3 title "Speed-up over OMP SIMD Scan + GCC",\
     "./plot.dat" using 1:11 with linespoints lw 3 title "Speed-up over Baseline + Clang",\
     "./plot.dat" using 1:12 with linespoints lw 3 title "Speed-up over OMP SIMD Scan + Clang",\
     "./plot.dat" using 1:13 with linespoints lw 3 title "Speed-up over Baseline + ICC",\
     "./plot.dat" using 1:14 with linespoints lw 3 title "Speed-up over OMP SIMD Scan + ICC"
EOF
