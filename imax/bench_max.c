#define _POSIX_C_SOURCE  200112L
#include "stdlib.h"
#include "stdio.h"
#include "immintrin.h"
#include "mkl.h"
#include "math.h"
#include "float.h"
#include "limits.h"

#define N_UNROLL          (64*2)
#define NUM_ELEMS_IN_REG  (16)
#define NTRIALS           (100)
#define SEED              (42)

#include "max_ref.cxx"
#if defined (USE_MAX_IDX_TRACKING)
#include "max_idx_tracking.cxx"
#elif defined (USE_MAX_BLK_TRACKING)
#include "max_block_tracking.cxx"
#endif

/* extern double my_dsecnd(); */

typedef enum {
  ASCEDNING_ORDER=0,
  DESCENDING_ORDER=1,
} data_order_t;

void init_x (float *x, int n, data_order_t order)
{
  if (order == ASCEDNING_ORDER) {
    for (int i=0; i<n; i++) {
      x[i] = i;
    }
  } else if (order == DESCENDING_ORDER) {
    for (int i=0; i<n; i++) {
      x[i] = n-i;
    }
  } else {
    srand(order);
    for (int i=0; i<n; i++) {
      x[i] = rand()/((float) RAND_MAX - 0.5);
    }
  }
}

int main (int argc, char **argv)
{
  double t_iter_start[NTRIALS];
  double t_iter_end[NTRIALS];
  float *x = NULL;
  data_order_t order;

  if (argc < 4) {
    printf ("usage : %s <size, int>   <data-order, 0|1|int>   <block-size, int>\n\n", argv[0]); 
    printf ("\tsize -> Integer specifying the number of elements in input vector\n"
            "\tdata-order -> Integer specifying on how-to initialize the input vector\n"
            "\t\t0 -> vector is initialized with values in ascending order\n"
            "\t\t1 -> vector is initialized with values in descending order\n"
            "\t\tany other integer -> the specified integer is used as random seed to fill entries\n"
            "\tblock-size -> Used only in AVX512 blocking implementation\n"
            );
    exit(1);
  }

  int n = atoi(argv[1]);
  if (argv[2]) {
    order = atoi(argv[2]);
  } else {
    order = SEED;
  }
  int block_size = atoi(argv[3]);
  int mkl_idx, tracking_idx, tracking_block, ref_idx, incx = 1;

  /* x = (float *)mkl_malloc(sizeof(float)*n, 64); */
  posix_memalign((void **)&x, 64, sizeof(float)*n); 

  init_x(x, n, order);

  // warm-up/validation
  ref_idx = ref_max(&n, x);
  // MKL returns Fortran-based index which starts from 1,
  // so we increment the C-based index for validation
  ref_idx++;

#if defined (USE_MAX_IDX_TRACKING)
  tracking_idx = max_idx_tracking(&n, x);
  tracking_idx++;
  if ((tracking_idx == ref_idx)) {
#elif defined (USE_MAX_BLK_TRACKING)
  tracking_block = max_block_tracking(&n, &block_size, x);
  tracking_block++;
  if ((tracking_block == ref_idx)) {
#elif defined (USE_MAX_MKL)
  mkl_idx = isamax(&n, x, &incx);
  if ((ref_idx == mkl_idx)) {
#else
  if (1) {
#endif
    printf ("validation passed\n");
  } else {
    printf ("validation failed\n");
    exit(1);
  }
  fflush(0);

  double t_start = dsecnd();
  t_start = dsecnd();
  t_start = dsecnd();
  for (int t=0; t<NTRIALS; t++) {
    t_iter_start[t] = dsecnd();

#if defined (USE_MAX_MKL)
    mkl_idx = isamax(&n, x, &incx);
#elif defined (USE_MAX_IDX_TRACKING)
    tracking_idx = max_idx_tracking(&n, x);
#elif defined (USE_MAX_BLK_TRACKING)
    tracking_block = max_block_tracking(&n, &block_size, x);
#elif defined (USE_MAX_REF)
    ref_idx = ref_max(&n, x);
#endif

    t_iter_end[t] = dsecnd();
  }
  double t_avg = (dsecnd() - t_start)/NTRIALS;

  tracking_idx++;
  ref_idx++;
  tracking_block++;

  double t_iter, t_best = FLT_MAX;
  for (int t=0; t<NTRIALS; t+=10) {
    for (int tt=t; tt<t+10; tt++) {
      t_iter = (t_iter_end[tt]-t_iter_start[tt]);
      if (t_best > t_iter) {
        t_best = t_iter;
      }
      /* printf (" %.2f ", t_iter*1.e6); fflush(0); */
    }
    /* printf ("\n"); */
  }
  /* printf ("t_best = %.2f\n", t_best*1.e6); fflush(0); */


#if defined (USE_MAX_MKL)
  printf ("Perf: n = %d, mkl_idx = %d, t_avg = %.2f, t_best = %.2f\n", n, mkl_idx, t_avg*1.e6, t_best*1.e6);
#elif defined (USE_MAX_IDX_TRACKING)
  printf ("Perf: n = %d, opt_idx = %d, t_avg = %.2f, t_best = %.2f\n", n, tracking_idx, t_avg*1.e6, t_best*1.e6);
#elif defined (USE_MAX_BLK_TRACKING)
  printf ("Perf: n = %d, block_size = %d, opt_idx = %d, t_avg = %.2f, t_best = %.2f\n", n, block_size, tracking_block, t_avg*1.e6, t_best*1.e6);
#elif defined (USE_MAX_REF)
  printf ("Perf: n = %d, ref_idx = %d, t_avg = %.2f, t_best = %.2f\n", n, ref_idx, t_avg*1.e6, t_best*1.e6);
#endif
  fflush(0);

  free(x);

  return 0;
}
