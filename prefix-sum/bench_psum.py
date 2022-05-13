#!/usr/bin/python

import numpy as np
import sys
import time
import math
import ctypes
import os
from numpy.ctypeslib import ndpointer

start_sz = 1024
end_sz = 102400+1024
step_sz = 1024
ntrials = 50
err_tol = 1e-5

print("Python:")
print(sys.version)
print("Numpy: {}".format(np.__version__))

os.environ["OMP_PLACES"] = "cores"
os.environ["OMP_PROC_BIND"] = "close"

print("OMP_PLACES is set to:", os.environ["OMP_PLACES"])
print("OMP_PROC_BIND is set to:", os.environ["OMP_PROC_BIND"])

# libavx512_psum.so provide OpenMP threaded avx512_psum() function with interface:
# void parallel_avx512_psum(int *n, double *src, double *dst, double *alpha);
avx512_psum_lib  = ctypes.cdll.LoadLibrary("./lib_par_avx512_psum.so")
avx512_psum_func = avx512_psum_lib.parallel_avx512_psum
avx512_psum_func.restype = None
avx512_psum_func.argtypes = [ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                         ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                         ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                         ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")]

n = np.zeros(1, dtype=np.int32)
alpha = np.zeros(1, dtype=np.float64)

# src = np.full(end_sz, 3.0, dtype=np.float64)
src = np.random.random_sample(end_sz).astype(np.float64)
dst_np = np.zeros(end_sz, dtype=np.float64)
dst_avx512 = np.zeros(end_sz, dtype=np.float64)

t_iter_np = np.zeros(ntrials)
t_iter_avx512 = np.zeros(ntrials)

# Find overhead of calling numpy and shared lib function
src_x = src[0:1]
dst_np_x = dst_np[0:1]
dst_avx512_x = dst_avx512[0:1]
n[0] = 1
for t in range(0, ntrials):
    t_start = time.time()
    dst_np_x = np.cumsum(src_x)
    t_end = time.time()
    t_iter_np[t] = int(round((t_end - t_start) * pow(10,9)))

    t_start = time.time()
    avx512_psum_func(n, src_x, dst_avx512_x, alpha)
    t_end = time.time()
    t_iter_avx512[t] = int(round((t_end - t_start) * pow(10,9)))

t_best_np = t_iter_np[0]
for t in t_iter_np:
    if t_best_np > t:
        t_best_np = t

t_best_avx512 = t_iter_avx512[0]
for t in t_iter_avx512:
    if t_best_avx512 > t:
        t_best_avx512 = t

t_overhead_np = t_best_np
t_overhead_avx512 = t_best_avx512
print("\nMeasuring overhead with n=1, "
"overheads(in ns) for numpy = {}, C-func from shared-lib = {}".format(t_overhead_np, t_overhead_avx512))

# Start sweep
for sz in range(start_sz, end_sz, step_sz):
    src_x = src[0:sz]
    dst_np_x = dst_np[0:sz]
    dst_avx512_x = dst_avx512[0:sz]
    n[0] = src_x.size

    for t in range(0, ntrials):
        t_start = time.time()
        dst_np_x = np.cumsum(src_x)
        t_end = time.time()
        t_iter_np[t] = int(round((t_end - t_start) * pow(10,9)))

        t_start = time.time()
        avx512_psum_func(n, src_x, dst_avx512_x, alpha)
        t_end = time.time()
        t_iter_avx512[t] = int(round((t_end - t_start) * pow(10,9)))

    for i in range(0,n[0]):
        if (math.fabs(dst_np_x[i] - dst_avx512_x[i]) > err_tol):
            print("n = {} validation failed at index = {}. Expected = {}, Observed = {}".format(sz, i, dst_np_x[i], dst_avx512_x[i]))
            sys.exit("quitting!")

    t_best_np = t_iter_np[0]
    for t in t_iter_np:
        if t_best_np > t:
            t_best_np = t

    t_best_avx512 = t_iter_avx512[0]
    for t in t_iter_avx512:
        if t_best_avx512 > t:
            t_best_avx512 = t

    t_best_np = t_best_np - t_overhead_np
    t_best_avx512 = t_best_avx512 - t_overhead_avx512
    if (t_best_np > 0 and t_best_avx512 > 0):
        speed_up = t_best_np/t_best_avx512
    else:
        print("WARNING: n = {} unreliable timing/too much overheads..".format(sz))
        speed_up = 1

    print("perf(ns): n = {}, np-psum = {}, avx512-psum = {}, speed-up = {:.2f}".format(sz, t_best_np, t_best_avx512, speed_up))

print("validation passed for all sizes")
