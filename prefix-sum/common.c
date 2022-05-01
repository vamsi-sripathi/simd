#include "stdio.h"
#include "stdlib.h"
#include "limits.h"
#include "float.h"
#include "math.h"
#define ERR_TOL           (1.e-7)

void compute_stats(int ntrials, double *t_iter, double *t_total, double *t_best, double *t_worst)
{
  double total = 0., worst = FLT_MIN, best = FLT_MAX;

  for (int t=0; t<ntrials; t++) {
    total += t_iter[t];

    if (best > t_iter[t]) {
      best = t_iter[t];
    }

    if (t_iter[t] > worst) {
      worst = t_iter[t];
    }

#ifdef VERBOSE
    printf ("trial-%d, time(ns) = %.2f\n", t, t_iter[t]*1.e9);
    fflush(0);
#endif
  }

  *t_total = total;
  *t_best  = best;
  *t_worst = worst;
}

int check_results (int *p_n, double *ref, double *obs)
{
  int n = *p_n;
  for (int i=0; i<n; i++) {
    if (fabs(ref[i]-obs[i]) > ERR_TOL) {
      printf ("ERROR: Index = %d, Expected = %e, Observed = %e, Abs_diff = %e\n", i, ref[i], obs[i], fabs(ref[i]-obs[i]));
      return 1;
    }
  }
  return 0;
}

void init_x (double *x, int n)
{
  srand(42);
  for (int i=0; i<n; i++) {
    x[i] = rand()/((double) RAND_MAX - 0.5);
  }
}


