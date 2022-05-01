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

  if (n_tail & 8) {
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

    n_tail -= 8;
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
