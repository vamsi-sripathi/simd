#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "math.h"
#include "mkl.h"
#include <sys/time.h>

#define NTRIALS           (100)
#define PAGE_ALIGN        (2ULL << 20)
#define AVX512_ALIGN      (64)

#define STRINGIFY(x) STRINGIFY_(x)
#define STRINGIFY_(x)  #x

#if defined (USE_REF)
#define OPT_FNAME  ref_psum
#elif defined (USE_AVX512)
#define OPT_FNAME  avx512_psum
#elif defined (USE_OMP)
#define OPT_FNAME  omp_psum
#else
#error "one of USE_REF|USE_AVX512|USE_OMP should be defined"
#endif

#define REF_FNAME  ref_psum
#define TIMER      dsecnd

// For GCC
/* #define TIMER      __rdtsc */
/* #include "mm_malloc.h" */
/* #include <x86intrin.h> */
// End for GCC

void init_x (double *x, int n);
int check_results (int *p_n, double *ref, double *obs);
void compute_stats (int ntrials, double *t_iter, double *t_total, double *t_best, double *t_worst);

void ref_psum (int *p_n, double *restrict src, double *restrict dst, double *p_init_val);
void omp_psum (int *p_n, double *restrict src, double *restrict dst, double *p_init_val);
void avx512_psum (int *p_n, double *restrict src, double *restrict dst, double *p_init_val);

double timer_us() {
	struct timeval mytime;

	gettimeofday( &mytime, (struct timezone *)0);

	return (double)(mytime.tv_sec) * 1.e6 + (double)(mytime.tv_usec);
}

int main (int argc, char **argv)
{
  if (argc < 4) {
    printf ("Benchmark to measure prefix-sum operation for different input vector sizes.\n"
            "USAGE: %s <start-size> <end-size> <step-size>\n", argv[0]); 
    printf ("\t{start, end, step}-size ->  Integers controlling the size of inputs used in the benchmark\n"); 
    exit(1);
  }

  int start = atoi(argv[1]);
  int end   = atoi(argv[2]);
  int step  = atoi(argv[3]);

  int n = end;
  double *x = NULL, *y = NULL, *y_ref = NULL;
  double t_iter_start[NTRIALS], t_iter_end[NTRIALS], t_iter[NTRIALS];
  double t_total, t_best, t_worst;
  double init_value = 0.;

#if 0
  x     = (double *)_mm_malloc(sizeof(double)*n, PAGE_ALIGN);
  y     = (double *)_mm_malloc(sizeof(double)*n, PAGE_ALIGN);
  y_ref = (double *)_mm_malloc(sizeof(double)*n, PAGE_ALIGN);
#else
  posix_memalign((void **)&x, PAGE_ALIGN, sizeof(double)*n);
  posix_memalign((void **)&y, PAGE_ALIGN, sizeof(double)*n);
  posix_memalign((void **)&y_ref, PAGE_ALIGN, sizeof(double)*n);
#endif

  printf ("INFO: Benchmarking %s ..\n", STRINGIFY(OPT_FNAME));

  //timer warm-up
  t_iter_start[0] = TIMER();
  t_iter_start[0] = TIMER();

  for (int s=start; s<=end; s+=step) {
    for (int t=0; t<NTRIALS; t++) {
      init_x(x, n);

      if (t == 0) {
        OPT_FNAME(&s, x, y, &init_value);
        REF_FNAME(&s, x, y_ref, &init_value);
        if (check_results(&s, y_ref, y)) {
          printf ("INFO: n = %d validation failed..skipping perf benchmarking\n", s); 
          fflush(0);
          exit(1);
        }
      } else {
        t_iter_start[t-1] = TIMER();
        OPT_FNAME(&s, x, y, &init_value);
        t_iter_end[t-1] = TIMER();
        t_iter[t-1] = t_iter_end[t-1] - t_iter_start[t-1];
      }
    }

    compute_stats(NTRIALS-1, t_iter, &t_total, &t_best, &t_worst);

    printf ("perf(ns): n = %d, worst = %.2f, avg = %.2f, best = %.2f\n",
            s, t_worst*1.e9, (t_total*1.e9)/(NTRIALS-1), t_best*1.e9);
    fflush(0);
  }

#if 0
  _mm_free(x);
  _mm_free(y);
  _mm_free(y_ref);
#else
  free(x);
  free(y);
  free(y_ref);
#endif

  return 0;
}
