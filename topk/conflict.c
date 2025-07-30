#include "immintrin.h"
#include "stdio.h"

int main (int argc, char **argv)
{
  int x[] = {7, 0, 1, 7, 7, 5, 5, 0};

  int res_x[8], res_conflicts[8], res_lzcnt[8], res_y[8];

  __m256i ymm_x = _mm256_loadu_epi32(x);
  __m256i conflicts = _mm256_conflict_epi32(ymm_x);
  __m256i lzcnt = _mm256_lzcnt_epi32(conflicts);
  __m256i tmp_const = _mm256_set1_epi32(31);
  __m256i ymm_y = _mm256_sub_epi32(tmp_const, lzcnt);

  _mm256_storeu_epi32(res_x, ymm_x);
  _mm256_storeu_epi32(res_conflicts, conflicts);
  _mm256_storeu_epi32(res_lzcnt, lzcnt);
  _mm256_storeu_epi32(res_y, ymm_y);

  for (int i=0; i<8; i++) {
    printf ("%d ", res_x[i]);
  }
  printf ("\n");
  for (int i=0; i<8; i++) {
    printf ("%d ", res_conflicts[i]);
  }
  printf ("\n");
  for (int i=0; i<8; i++) {
    printf ("%d ", res_lzcnt[i]);
  }
  printf ("\n");
  for (int i=0; i<8; i++) {
    printf ("%d ", res_y[i]);
  }
  printf ("\n");

  return 0;
}
