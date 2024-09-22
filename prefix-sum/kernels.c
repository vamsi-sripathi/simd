void ref_psum (int *p_n, double *restrict src, double *restrict dst, double *p_init_val)
{
  int n = *p_n;
	double tmp = *p_init_val;

	for (int i=0; i<n; i++) {
		tmp += src[i];
		dst[i] = tmp;
	}
}

#if defined (USE_OMP)
void omp_psum (int *p_n, double *restrict src, double *restrict dst, double *p_init_val)
{
  int n = *p_n;
  double tmp = *p_init_val;

#pragma omp simd reduction(inscan, +:tmp)
    for(int i=0; i<n; i++){
      tmp += src[i];
#pragma omp scan inclusive(tmp)
      dst[i] = tmp;
    }
}
#endif

#if defined (USE_AVX512)
#include "immintrin.h"

#define N_UNROLL          (16)
#define NUM_ELES_IN_ZMM   (8)
#define OPT1

void avx512_psum (int *p_n, double *restrict src, double *restrict dst, double *p_init_val)
{
  __m512d zmm0, zmm1, zmm2, zmm3, zmm4, zmm_acc, zmm_tmp1 ,zmm_tmp2;
  __m512d zmm5, zmm6, zmm7, zmm8, zmm9, zmm10, zmm11;
  __m512i idx, acc_idx;

  int n = *p_n;

  zmm_acc = _mm512_set1_pd(*p_init_val);
  acc_idx = _mm512_set1_epi64(7);
  idx     = _mm512_set1_epi64(3);


  int n_block = (n/N_UNROLL)*N_UNROLL;
  int n_tail  = n - n_block;

  for (int j=0; j<n_block; j+=N_UNROLL) {
    zmm0 = _mm512_loadu_pd(src);
    zmm5 = _mm512_loadu_pd(src+NUM_ELES_IN_ZMM);

    zmm2  = _mm512_maskz_permute_pd(0xAA, zmm0, 0x00);
    zmm7  = _mm512_maskz_permute_pd(0xAA, zmm5, 0x00);
    zmm10 = _mm512_add_pd(zmm0, zmm2);
    zmm11 = _mm512_add_pd(zmm5, zmm7);

#if defined (OPT1)
    zmm1  = _mm512_maskz_permutex_pd(0xCC, zmm0, 0x40);
    zmm6  = _mm512_maskz_permutex_pd(0xCC, zmm5, 0x40);
    zmm10 = _mm512_add_pd(zmm10, zmm1);
    zmm11 = _mm512_add_pd(zmm11, zmm6);

    zmm1  = _mm512_maskz_permute_pd(0xCC, zmm1, 0x44);
    zmm6  = _mm512_maskz_permute_pd(0xCC, zmm6, 0x44);
#elif defined (OPT2)
    zmm1  = _mm512_maskz_permutex_pd(0xCC, zmm10, 0x50);
    zmm6  = _mm512_maskz_permutex_pd(0xCC, zmm11, 0x50);
#endif
    zmm10 = _mm512_add_pd(zmm10, zmm1);
    zmm11 = _mm512_add_pd(zmm11, zmm6);

    zmm_tmp1 = _mm512_maskz_permutexvar_pd(0xF0, idx, zmm10);
    zmm_tmp2 = _mm512_maskz_permutexvar_pd(0xF0, idx, zmm11);
    zmm10    = _mm512_add_pd(zmm10, zmm_tmp1);
    zmm11    = _mm512_add_pd(zmm11, zmm_tmp2);

    zmm_tmp1 = _mm512_add_pd(zmm10, zmm_acc);
    zmm_acc  = _mm512_add_pd(zmm11, zmm_tmp1);
    zmm_acc  = _mm512_permutexvar_pd(acc_idx, zmm_acc);

    zmm_tmp2 = _mm512_permutexvar_pd(acc_idx, zmm_tmp1);
    _mm512_storeu_pd(dst, zmm_tmp1);

    zmm11 = _mm512_add_pd(zmm11, zmm_tmp2);
    _mm512_storeu_pd(dst+NUM_ELES_IN_ZMM, zmm11);

    src += N_UNROLL;
    dst += N_UNROLL;
  }

  if (n_tail & NUM_ELES_IN_ZMM) {
    zmm0  = _mm512_loadu_pd(src);

    zmm2  = _mm512_maskz_permute_pd(0xAA, zmm0, 0x00);
    zmm10 = _mm512_add_pd(zmm0, zmm2);

    zmm1  = _mm512_maskz_permutex_pd(0xCC, zmm0, 0x40);
    zmm10 = _mm512_add_pd(zmm10, zmm1);

    zmm1  = _mm512_maskz_permute_pd(0xCC, zmm1, 0x44);
    zmm10 = _mm512_add_pd(zmm10, zmm1);

    zmm_tmp1 = _mm512_maskz_permutexvar_pd(0xF0, idx, zmm10);
    zmm10    = _mm512_add_pd(zmm10, zmm_tmp1);

    zmm_tmp1 = _mm512_add_pd(zmm10, zmm_acc);
    zmm_acc  = _mm512_permutexvar_pd(acc_idx, zmm_tmp1);

    _mm512_storeu_pd(dst, zmm_tmp1);

    src  += NUM_ELES_IN_ZMM;
    dst  += NUM_ELES_IN_ZMM;

    n_tail -= NUM_ELES_IN_ZMM;
  }

  if (n_tail) {
    __mmask8 k1 = (1<<n_tail)-1;
    zmm0  = _mm512_maskz_loadu_pd(k1, src);

    zmm2  = _mm512_maskz_permute_pd(0xAA, zmm0, 0x00);
    zmm10 = _mm512_add_pd(zmm0, zmm2);

    zmm1  = _mm512_maskz_permutex_pd(0xCC, zmm0, 0x40);
    zmm10 = _mm512_add_pd(zmm10, zmm1);

    zmm1  = _mm512_maskz_permute_pd(0xCC, zmm1, 0x44);
    zmm10 = _mm512_add_pd(zmm10, zmm1);

    zmm_tmp1 = _mm512_maskz_permutexvar_pd(0xF0, idx, zmm10);
    zmm10    = _mm512_add_pd(zmm10, zmm_tmp1);

    zmm10 = _mm512_add_pd(zmm10, zmm_acc);

    _mm512_mask_storeu_pd(dst, k1, zmm10);
  }
}
#endif

#if defined (USE_THREADS)
#include "omp.h"
#define MAX_NTHRS         (4096)

static void update (int *p_n, double *restrict p_dst, double *p_alpha)
{
  int n = *p_n;
  double alpha = *p_alpha;

  int n_block = (n/N_UNROLL)*N_UNROLL;
  int n_tail  = n - n_block;

  __m512d zmm0, zmm1, zmm_alpha;
  zmm_alpha = _mm512_broadcastsd_pd(_mm_load_sd(p_alpha));

  for (int i=0; i<n_block; i+=N_UNROLL) {
    zmm0 = _mm512_loadu_pd(p_dst);
    zmm1 = _mm512_loadu_pd(p_dst+NUM_ELES_IN_ZMM);

    zmm0 = _mm512_add_pd(zmm0, zmm_alpha);
    zmm1 = _mm512_add_pd(zmm1, zmm_alpha);

    _mm512_storeu_pd(p_dst, zmm0);
    _mm512_storeu_pd(p_dst+NUM_ELES_IN_ZMM, zmm1);

    p_dst += N_UNROLL;
  }

  if (n_tail & NUM_ELES_IN_ZMM) {
    zmm0 = _mm512_loadu_pd(p_dst);
    zmm0 = _mm512_add_pd(zmm0, zmm_alpha);
    _mm512_storeu_pd(p_dst, zmm0);

    p_dst  += NUM_ELES_IN_ZMM;
    n_tail -= NUM_ELES_IN_ZMM;
  }

  if (n_tail) {
    __mmask8 k1 = (1<<n_tail)-1;
    zmm0 = _mm512_maskz_loadu_pd(k1, p_dst);
    zmm0 = _mm512_add_pd(zmm0, zmm_alpha);
    _mm512_mask_storeu_pd(p_dst, k1, zmm0);
  }
}


void parallel_avx512_psum (int *p_n, double *restrict p_src, double *restrict p_dst, double *p_init_val)
{
  int n = *p_n;

  if (n <= 4096) {
    return avx512_psum(p_n, p_src, p_dst, p_init_val);
  }

#if 0
  int nthrs = n/8192;
  if (n%8192) {
    nthrs++;
  }
  if (nthrs > 48) {
    nthrs = 48;
  }
#endif

  double *partial_psum = (double *)_mm_malloc(sizeof(double)*MAX_NTHRS, 64);

#pragma omp parallel default(shared) //num_threads(nthrs)
  {
  int nthrs = omp_get_num_threads();
  int ithr  = omp_get_thread_num();

  int n_chunk, n_tail, offset;
#if 1
  int n_tail_blk, nthrs_tail;

  n_chunk    = (n/nthrs/8)*8;
  n_tail_blk = n - n_chunk*nthrs;
  nthrs_tail = n_tail_blk/8;
  n_tail     = n - n_chunk*nthrs - 8*nthrs_tail;

  offset   = ithr*n_chunk;
  if (ithr) {
    offset += n_tail + ((ithr < nthrs_tail) ? ithr*8 : nthrs_tail*8);
  }

  n_chunk += ((ithr < nthrs_tail) ? 8 : 0);
  if (ithr == 0 && n_tail) {
    n_chunk += n_tail;
  }
#else
  n_chunk = n/nthrs;
  n_tail  = n - n_chunk*nthrs;
  offset  = ((ithr == 0) ? 0 : ithr*n_chunk + n_tail);

  if (ithr == 0 && n_tail) {
    n_chunk += n_tail;
  }
#endif
  /* printf("[%d] chunk = %d, offset = %d\n", ithr, n_chunk, offset); fflush(0); */

  avx512_psum(&n_chunk, p_src+offset, p_dst+offset, p_init_val);

  partial_psum[ithr] = *(p_dst+offset+n_chunk-1);
#pragma omp barrier

  double acc = 0.;
  for (int i=0; i<ithr; i++) {
    acc += partial_psum[i];
  }

  update(&n_chunk, p_dst+offset, &acc);

  } // end parallel region

  _mm_free(partial_psum);

}
#endif
