#include "immintrin.h"
#include "stdio.h"
#include "stdlib.h"

void print_f32ymm(__m256 ymm, char *s)
{
  float buf[8];
  _mm256_storeu_ps(buf, ymm);
  printf ("%s: ",s);
  for (int i=0; i<8; i++) {
    printf ("%.2f ", buf[i]);
  }
  printf ("\n");
  fflush(0);
}

void print_i32ymm(__m256i ymm, char *s)
{
  int buf[8];
  _mm256_storeu_epi32(buf, ymm);
  printf ("%s: ",s);
  for (int i=0; i<8; i++) {
    printf ("%d ", buf[i]);
  }
  printf ("\n");
  fflush(0);
}


void ref_topk (int *p_n, float *topk, float *x)
{
  int n = *p_n;

  float t;
  // lowest value at index-0
  for (int i=0; i<n; i++) {
    for (int j=i+1; j<n; j++) {
      if (x[j] < x[i]) {
        t = x[j];
        x[j] = x[i];
        x[i] = t;
      }
    }
  }

  for (int i=n-8; i<n; i++) {
    *topk = x[i];
    topk++;
  }
}

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

void my_topk (int *p_n, float *topk, float *y)
{
  __m256    curr_topk, ymm_y, ymm_tmp, topk_lowest, topk_highest;
  __m256    y0, y1, y2, y3, y4, y5, y6, y7;
  __m256i   indices, ymm_indices;
  __mmask8  k0, k1, k2, k3, k4, k5, k6, k7;
  __m128    xmm0, xmm1, xmm2, xmm3;
  unsigned char all_ones;

  int n = *p_n;
  int tmp[8];
  float buffer[8];
  int permute_indices[8];
  float new_topk[8];
  int step = 8;

  curr_topk = _mm256_loadu_ps(topk);

  /* print_f32ymm(curr_topk, "curr_topk"); */

  for (int c=0; c<n; c+=8) {
    ymm_y = _mm256_loadu_ps(&y[c]);
    /* print_f32ymm(ymm_y, "new_input"); */

    // quick checks
    // if all incoming values is less than the lowest topk
    topk_lowest = _mm256_broadcastss_ps(_mm256_castps256_ps128(curr_topk));
    k0 = _mm256_cmp_ps_mask(ymm_y, topk_lowest, _CMP_LT_OS);
    // if k0 has all ones ie all incoming values are less than the lowest topk, skip iteration
    _kortest_mask8_u8(k0, k0, &all_ones);
    if (all_ones) {
      /* printf ("\n quick return.. skipping iteration..\n"); */
      continue;
    }

    topk_highest = _mm256_broadcastss_ps(_mm_permute_ps(_mm256_extractf32x4_ps(curr_topk, 0x1), 0x3));
    k1 = _mm256_cmp_ps_mask(topk_highest, ymm_y, _CMP_LT_OS);
    // if k1 has all ones ie the highest topk is less than all incoming values,
    // we need to sort the incoming values and skip iteration
    _kortest_mask8_u8(k1, k1, &all_ones);
    if (all_ones) {
      /* printf ("\n sort all and skipping iteration..\n"); */
      ref_topk(&step, buffer, y);
      curr_topk = _mm256_loadu_ps(buffer);
      continue;
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
    for (int kk=0; kk<8; kk++) {
      tmp[kk] = -1;
    }

    _mm256_set1_epi32(-1);
    k0 = _mm256_cmp_ps_mask(y0, curr_topk, _CMP_GT_OS);
    _mm256_mask_set1_epi32(indices, k0, _bit_scan_reverse(k0));

    if (k0) {
      tmp[0]  = _bit_scan_reverse(k0);
    }

    k1 = _mm256_cmp_ps_mask(y1, curr_topk, _CMP_GT_OS);
    if (k1) {
      tmp[1]  = _bit_scan_reverse(k1);
    }

    k2 = _mm256_cmp_ps_mask(y2, curr_topk, _CMP_GT_OS);
    if (k2) {
      tmp[2]  = _bit_scan_reverse(k2);
    }

    k3 = _mm256_cmp_ps_mask(y3, curr_topk, _CMP_GT_OS);
    if (k3) {
      tmp[3]  = _bit_scan_reverse(k3);
    }

    k4 = _mm256_cmp_ps_mask(y4, curr_topk, _CMP_GT_OS);
    if (k4) {
      tmp[4]  = _bit_scan_reverse(k4);
    }

    k5 = _mm256_cmp_ps_mask(y5, curr_topk, _CMP_GT_OS);
    if (k5) {
      tmp[5]  = _bit_scan_reverse(k5);
    }

    k6 = _mm256_cmp_ps_mask(y6, curr_topk, _CMP_GT_OS);
    if (k6) {
      tmp[6]  = _bit_scan_reverse(k6);
    }

    k7 = _mm256_cmp_ps_mask(y7, curr_topk, _CMP_GT_OS);
    if (k7) {
      tmp[7]  = _bit_scan_reverse(k7);
    }
    indices = _mm256_loadu_epi32(tmp);
    /* print_i32ymm(indices, "cmp_hibits"); */

    int count = 0, set_count, new_entries = 0;
    int idx = 7;
    for (int i=7; i>=0 && count<8; i--) {
      k0 = _mm256_cmp_epi32_mask(indices, _mm256_set1_epi32(i), _MM_CMPINT_EQ);
      set_count = _popcnt32(k0);
      if (set_count) {
        count += set_count;
        y0 = _mm256_maskz_compress_ps(k0, ymm_y);
        sort_register(y0, set_count, &buffer[new_entries]);
        while (set_count) {
          permute_indices[idx--] = new_entries++;
          set_count--;
        }
      }
      permute_indices[idx--] = 8+i;
      count++;
    }

    ymm_tmp = _mm256_loadu_ps(buffer);
    ymm_indices = _mm256_loadu_epi32(permute_indices);
    curr_topk = _mm256_permutex2var_ps(ymm_tmp, ymm_indices, curr_topk);
    /* print_i32ymm(ymm_indices, "permute_indices"); */
    /* print_f32ymm(curr_topk, "new_topk"); */
  }

  _mm256_storeu_ps(topk, curr_topk);
}

int main (int argc, char **argv)
{
  if (argc != 2) {
    printf ("usage: %s <n>\n", argv[0]);
    exit(1);
  }

  int n = atoi(argv[1]);
  
  float *x = (float *)_mm_malloc(sizeof(float)*n, 64);
  float ref_top8[8], opt_top8[8];
  float t;

#ifdef DEBUG
  printf ("input:\n");
  srand(2);
  for (int i=0; i<n; i++) {
    x[i] = rand()%8192;
    printf ("%.1f\n", x[i]);
  }
  printf ("\n");
#endif

  int step = 8;
  ref_topk(&step, opt_top8, x);

#ifdef DEBUG
  printf ("initial sorted list: ");
  for (int i=0; i<8; i++) {
    printf ("%.1f ", opt_top8[i]);
  }
  printf ("\n");
#endif
  
  int nn = n-step;
  my_topk(&nn, opt_top8, x+8);

  ref_topk(&n, ref_top8, x);  

  printf ("ref_topk: ");
  for (int i=0; i<8; i++) {
    printf ("%.1f ", ref_top8[i]);
  }
  printf ("\n");

  printf ("opt_topk: ");
  for (int i=0; i<8; i++) {
    printf ("%.1f ", opt_top8[i]);
  }
  printf ("\n");

  for (int i=0; i<8; i++) {
    printf ("%.1f ", ref_top8[i]);

    if (ref_top8[i] != opt_top8[i]) {
      printf ("validation failed at %d. Expected = %.1f, Observed = %.1f\n", i, ref_top8[i], opt_top8[i]);
      exit (1);
    }
  }
  printf ("\nvalidation passed\n");
  fflush(0);

  _mm_free(x);

  return 0;
}
