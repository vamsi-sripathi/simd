#include "immintrin.h"
#include "stdio.h"
#include "stdlib.h"


void sort_register (__m256 ymm, int count, float *buffer)
{
  // highest value should be at index-0
  _mm256_mask_storeu_ps(buffer, (1<<count)-1, ymm);

  float tmp;
  for (int i=0; i<count; i++) {
    for (int j=i+1; j<count; j++) {
      if (buffer[j] > buffer[i]) {
        tmp = buffer[i];
        buffer[i] = buffer[j];
        buffer[j] = tmp;
      }
    }
  }
}

void my_topk (float *x, float *y)
{
  __m256    ymm_x, ymm_y, ymm_new_topk, ymm_tmp, topk_lowest, topk_highest;
  __m256    y0, y1, y2, y3, y4, y5, y6, y7;
  __m256i   indices, ymm_indices;
  __mmask8  k0, k1, k2, k3, k4, k5, k6, k7;
  __m128    xmm0, xmm1, xmm2, xmm3;
  unsigned char all_ones;

  /* __m256i   zeros, ones, twos, threes, fours, fives, sixes, sevens; */
  /* zeros  = _mm256_setzero_si256(); */
  /* ones   = _mm256_set1_epi32(1); */
  /* twos   = _mm256_set1_epi32(2); */
  /* threes = _mm256_set1_epi32(3); */
  /* fours  = _mm256_set1_epi32(4); */
  /* fives  = _mm256_set1_epi32(5); */
  /* sixes  = _mm256_set1_epi32(6); */
  /* sevens = _mm256_set1_epi32(7); */

  int tmp[8];
  float buffer[8];
  int permute_indices[8];
  float new_topk[8];

  // loop
  ymm_x = _mm256_loadu_ps(x);
  ymm_y = _mm256_loadu_ps(y);

  // quick checks
  // if all incoming values is less than the lowest topk
  topk_lowest = _mm256_broadcastss_ps(_mm256_castps256_ps128(ymm_x));
  k0 = _mm256_cmp_ps_mask(ymm_y, topk_lowest, _CMP_LT_OS);
  // if k0 has all ones ie all incoming values are less than the lowest topk, skip iteration
  _kortest_mask8_u8(k0, k0, &all_ones);
  if (all_ones) {
    return;
  }

  topk_highest = _mm256_broadcastss_ps(_mm_permute_ps(_mm256_extractf32x4_ps(ymm_x, 0x1), 0x3));
  k1 = _mm256_cmp_ps_mask(topk_highest, ymm_y, _CMP_LT_OS);
  // if k1 has all ones ie the highest topk is less than all incoming values,
  // so we need to sort the incoming values and return
  _kortest_mask8_u8(k1, k1, &all_ones);
  if (all_ones) {
    sort_register(ymm_y, 8, buffer);
    // replace topk with buffer
  }

  xmm0 = _mm256_castps256_ps128(ymm_y);
  xmm1 = _mm_movehl_ps(xmm0, xmm0);
  y0 = _mm256_broadcastss_ps(xmm0);
  y1 = _mm256_broadcastss_ps(_mm_movehdup_ps(xmm0));
  y2 = _mm256_broadcastss_ps(xmm1);
  y3 = _mm256_broadcastss_ps(_mm_movehdup_ps(xmm1));

  xmm2 =_mm256_extractf32x4_ps(ymm_y, 0x1);
  xmm3 = _mm_movehl_ps(xmm2, xmm2);
  y4 = _mm256_broadcastss_ps(xmm2);
  y5 = _mm256_broadcastss_ps(_mm_movehdup_ps(xmm2));
  y6 = _mm256_broadcastss_ps(xmm3);
  y7 = _mm256_broadcastss_ps(_mm_movehdup_ps(xmm3));


  // if incoming element is not greater than any of current
  // topk, then it contains undefined value
  tmp[0]  = _bit_scan_reverse(_mm256_cmp_ps_mask(y0, ymm_x, _CMP_GT_OS));
  tmp[1]  = _bit_scan_reverse(_mm256_cmp_ps_mask(y1, ymm_x, _CMP_GT_OS));
  tmp[2]  = _bit_scan_reverse(_mm256_cmp_ps_mask(y2, ymm_x, _CMP_GT_OS));
  tmp[3]  = _bit_scan_reverse(_mm256_cmp_ps_mask(y3, ymm_x, _CMP_GT_OS));
  tmp[4]  = _bit_scan_reverse(_mm256_cmp_ps_mask(y4, ymm_x, _CMP_GT_OS));
  tmp[5]  = _bit_scan_reverse(_mm256_cmp_ps_mask(y5, ymm_x, _CMP_GT_OS));
  tmp[6]  = _bit_scan_reverse(_mm256_cmp_ps_mask(y6, ymm_x, _CMP_GT_OS));
  tmp[7]  = _bit_scan_reverse(_mm256_cmp_ps_mask(y7, ymm_x, _CMP_GT_OS));
  indices = _mm256_loadu_epi32(tmp);

  int count = 0, set_count, new_entries = 0;
  int idx = 7;
  for (int i=7; count<8; i--) {
    k0 = _mm256_cmp_epi32_mask(indices, _mm256_set1_epi32(i), _MM_CMPINT_EQ);
    set_count = _popcnt32(k0);
    if (set_count) {
      y0 = _mm256_maskz_compress_ps(k0, ymm_y);
      sort_register(y0, set_count, &buffer[new_entries]);
      while (set_count) {
        permute_indices[idx--] = new_entries++;
        set_count--;
      }
      count += set_count;
    }
    permute_indices[idx--] = 8+i;
    count++;
  }

  ymm_tmp = _mm256_loadu_ps(buffer);
  ymm_indices = _mm256_loadu_epi32(permute_indices);
  ymm_new_topk = _mm256_permutex2var_ps(ymm_tmp, ymm_indices, ymm_x);

  _mm256_storeu_ps(new_topk, ymm_new_topk);

  printf ("new_topk:\n");
  for (int i=0; i<8; i++) {
    printf ("%.1f ", new_topk[i]);
  }
  printf ("\n");

}

#if 0
  k7 = _mm256_cmp_epi32_mask(indices, sevens, _MM_CMPINT_EQ);
  k6 = _mm256_cmp_epi32_mask(indices, sixes, _MM_CMPINT_EQ);
  k5 = _mm256_cmp_epi32_mask(indices, fives, _MM_CMPINT_EQ);
  k4 = _mm256_cmp_epi32_mask(indices, fours, _MM_CMPINT_EQ);
  k3 = _mm256_cmp_epi32_mask(indices, threes, _MM_CMPINT_EQ);
  k2 = _mm256_cmp_epi32_mask(indices, twos, _MM_CMPINT_EQ);
  k1 = _mm256_cmp_epi32_mask(indices, ones, _MM_CMPINT_EQ);
  k0 = _mm256_cmp_epi32_mask(indices, zeros, _MM_CMPINT_EQ);


  count = _popcnt32(k7);
  if (count) {
    y7 = _mm256_maskz_compress_ps(k7, ymm_y);
    sort_register(y7, count);
  }

  if (count < 8) {
    count_k6 = _popcnt32(k6);
    if (count_k6) {
      y6 = _mm256_maskz_compress_ps(k6, ymm_y);
      sort_register(y6, count_k6);
      count += count_k6;
    }
  }

  if (count < 8) {
    count_k5 = _popcnt32(k5);
    if (count_k5) {
      y5 = _mm256_maskz_compress_ps(k5, ymm_y);
      sort_register(y5, count_k5);
      count += count_k5;
    }
  }

  if (count < 8) {
    count_k4 = _popcnt32(k4);
    if (count_k4) {
      y4 = _mm256_maskz_compress_ps(k4, ymm_y);
      sort_register(y4, count_k4);
      count += count_k4;
    }
  }

  if (count < 8) {
    count_k3 = _popcnt32(k3);
    if (count_k3) {
      y3 = _mm256_maskz_compress_ps(k3, ymm_y);
      sort_register(y3, count_k3);
      count += count_k3;
    }
  }

  if (count < 8) {
    count_k2 = _popcnt32(k2);
    if (count_k2) {
      y2 = _mm256_maskz_compress_ps(k2, ymm_y);
      sort_register(y2, count_k2);
      count += count_k2;
    }
  }

  if (count < 8) {
    count_k1 = _popcnt32(k1);
    if (count_k1) {
      y1 = _mm256_maskz_compress_ps(k1, ymm_y);
      sort_register(y1, count_k1);
      count += count_k1;
    }
  }

  if (count < 8) {
    count_k0 = _popcnt32(k0);
    if (count_k0) {
      y0 = _mm256_maskz_compress_ps(k0, ymm_y);
      sort_register(y0, count_k0);
      count += count_k0;
    }
  }

  /* float sets[8][8] = {0}; */

  /* _mm256_mask_compressstoreu_ps(sets[0], k0, ymm_y); */
  /* _mm256_mask_compressstoreu_ps(sets[1], k1, ymm_y); */
  /* _mm256_mask_compressstoreu_ps(sets[2], k2, ymm_y); */
  /* _mm256_mask_compressstoreu_ps(sets[3], k3, ymm_y); */
  /* _mm256_mask_compressstoreu_ps(sets[4], k4, ymm_y); */
  /* _mm256_mask_compressstoreu_ps(sets[5], k5, ymm_y); */
  /* _mm256_mask_compressstoreu_ps(sets[6], k6, ymm_y); */
  /* _mm256_mask_compressstoreu_ps(sets[7], k7, ymm_y); */

  /* printf ("Sets\n"); */
  /* for (int i=0; i<8; i++) { */
  /*   printf ("s[%d]: ",i); */
  /*   for (int j=0; j<8; j++) { */
  /*     printf ("%.1f ", sets[i][j]); */
  /*   } */
  /*   printf ("\n"); */
  /* } */
#endif

int main (int argc, char **argv)
{
  float sorted_input[] = {1., 2., 3., 4.,
                          5., 6., 7., 8.};
  float unsorted_input[] = {15., 3., 8., 17,
                            20., 2., 13., 5.};
  float res[8];

  float x[8], y[8], t;
  srand(2);
  for (int i=0; i<8; i++) {
   x[i] = rand()%100;
  }
  for (int i=0; i<8; i++) {
   y[i] = rand()%100;
  }
  y[7] = 5.;

  printf ("Input: ");
  for (int i=0; i<8; i++) {
    printf ("%.1f ", x[i]);
  }
  printf ("\n");

  for (int i=0; i<8; i++) {
    for (int j=i+1; j<8; j++) {
      if (x[j] < x[i]) {
        t = x[j];
        x[j] = x[i];
        x[i] = t;
      }
    }
  }
  
  printf ("Sorted Input: ");
  for (int i=0; i<8; i++) {
    printf ("%.1f ", x[i]);
  }
  printf ("\n");
  printf ("To sort: ");
  for (int i=0; i<8; i++) {
    printf ("%.1f ", y[i]);
  }
  printf ("\n");

  my_topk(x, y);

  return 0;
}
