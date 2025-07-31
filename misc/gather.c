#include "stdio.h"
#include "immintrin.h"
#include "stdlib.h"

#define ALIGN 64

int main(int argc, char **argv)
{
  int SIZE = atoi(argv[1]);

  double *p_output  = (double *)_mm_malloc(sizeof(double)*SIZE, ALIGN);
  double *p_input   = (double *)_mm_malloc(sizeof(double)*SIZE, ALIGN);
  int    *p_indices = (int *)_mm_malloc(sizeof(int)*SIZE, ALIGN);

  srand(42);

  for (int i=0; i<SIZE; i++) {
    p_input[i] = i;
    p_indices[i] = rand()%SIZE;
  }

#if 0
//#pragma vector nomultiple_gather_scatter_by_shuffles
  for (int i=0; i<SIZE; i++) {
    p_output[i] = p_input[p_indices[i]];
  }
#endif

#if 0
  __m256i ymm_indices;
  __m512d zmm_input;

  for (int i=0; i<SIZE; i+=8) {
    ymm_indices = _mm256_loadu_epi32(&p_indices[i]);
    zmm_input = _mm512_i32gather_pd (ymm_indices, p_input, _MM_SCALE_8);
    _mm512_storeu_pd(&p_output[i], zmm_input);
  }
#endif

  __m128d xmm0, xmm1, xmm2, xmm3;
  __m256d ymm0, ymm1;
  __m512d zmm0;

  for (int i=0; i<SIZE; i+=8) {
    xmm0 = _mm_load_sd(p_input + p_indices[0]);
    xmm0 = _mm_loadh_pd(xmm0, p_input + p_indices[0+1]);

    xmm1 = _mm_load_sd(p_input + p_indices[0+2]);
    xmm1 = _mm_loadh_pd(xmm1, p_input + p_indices[0+3]);

    xmm2 = _mm_load_sd(p_input + p_indices[0+4]);
    xmm2 = _mm_loadh_pd(xmm2, p_input + p_indices[0+5]);

    xmm3 = _mm_load_sd(p_input + p_indices[0+6]);
    xmm3 = _mm_loadh_pd(xmm3, p_input + p_indices[0+7]);

    ymm0 = _mm256_insertf128_pd(_mm256_castpd128_pd256(xmm0), xmm1, 0x1);
    ymm1 = _mm256_insertf128_pd(_mm256_castpd128_pd256(xmm2), xmm3, 0x1);

    zmm0 = _mm512_insertf64x4(_mm512_castpd256_pd512(ymm0), ymm1, 0x1);
    _mm512_storeu_pd(&p_output[i], zmm0);
    
    p_indices += 8;
  }

  p_indices -= SIZE;


  for (int i=0; i<SIZE; i++) {
    printf ("Index-%d = %d, Input-at-index = %.2f, Output-at-index = %.2f\n",
            i, p_indices[i], p_input[p_indices[i]], p_output[i]);

    if (p_output[i] != p_input[p_indices[i]]) {
      printf ("validation failed at index-%d\n", i);
      break;
    }
  }

  _mm_free(p_output);
  _mm_free(p_input);
  _mm_free(p_indices);

  return 0;
}
