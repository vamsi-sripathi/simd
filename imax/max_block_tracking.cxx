int max_block_tracking (int *p_n, int *p_block_size, float *p_x) 
{
  __m512 zmm_x0, zmm_x1, zmm_x2, zmm_x3, zmm_x4, zmm_x5, zmm_x6, zmm_x7;
  __m512 zmm_curr_max, zmm_tmp0, zmm_tmp1, zmm_tmp2, zmm_tmp3, zmm_tmp4, zmm_tmp5;
  __m256 ymm_x0, ymm_x1;
  __m128 xmm_x0, xmm_x1;
  __mmask16 m0;
  __m512 zmm_max      = _mm512_set1_ps(FLT_MIN);
  __m512i zmm_indices = _mm512_setzero_epi32();

  int n           = *p_n;
  int block_size  = *p_block_size;
  int num_blocks  = n/block_size;
  float block_max = FLT_MIN;
  int block_id    = 0;
  int block_start, offset;
  float curr_max;

  for (int k=0; k<num_blocks; k++) {
    for (int j=0; j<block_size; j+=N_UNROLL) {
      zmm_x0   = _mm512_loadu_ps(p_x);
      zmm_x1   = _mm512_loadu_ps(p_x+16);
      zmm_x2   = _mm512_loadu_ps(p_x+32);
      zmm_x3   = _mm512_loadu_ps(p_x+48);
      zmm_tmp0 = _mm512_max_ps(zmm_x0, zmm_x1);
      zmm_tmp1 = _mm512_max_ps(zmm_x2, zmm_x3);
#if N_UNROLL   > 64
      zmm_x4   = _mm512_loadu_ps(p_x+64);
      zmm_x5   = _mm512_loadu_ps(p_x+80);
      zmm_x6   = _mm512_loadu_ps(p_x+96);
      zmm_x7   = _mm512_loadu_ps(p_x+112);
      zmm_tmp2 = _mm512_max_ps(zmm_x4, zmm_x5);
      zmm_tmp3 = _mm512_max_ps(zmm_x6, zmm_x7);
      zmm_tmp4 = _mm512_max_ps(zmm_tmp0, zmm_tmp1);
      zmm_tmp5 = _mm512_max_ps(zmm_tmp2, zmm_tmp3);

      zmm_curr_max = _mm512_max_ps(zmm_tmp4, zmm_tmp5);
#else
      zmm_curr_max = _mm512_max_ps(zmm_tmp0, zmm_tmp1);
#endif
      zmm_max      = _mm512_max_ps(zmm_curr_max, zmm_max);

      p_x += N_UNROLL;
    }

    // reduction tree on zmm register to find max. value
    ymm_x0 = _mm512_castps512_ps256(zmm_max);
    ymm_x1 = _mm512_extractf32x8_ps(zmm_max, 1);
    ymm_x0 = _mm256_max_ps(ymm_x0, ymm_x1);

    xmm_x0 = _mm256_castps256_ps128(ymm_x0);
    xmm_x1 = _mm256_extractf32x4_ps(ymm_x0, 1);
    xmm_x0 = _mm_max_ps(xmm_x0, xmm_x1);

    xmm_x1 = _mm_permute_ps(xmm_x0, 0x0E);
    xmm_x0 = _mm_max_ps(xmm_x0, xmm_x1);

    xmm_x1 = _mm_permute_ps(xmm_x0, 0x01);
    xmm_x0 = _mm_max_ps(xmm_x0, xmm_x1);
    _mm_store_ss(&curr_max, xmm_x0);

    if (curr_max > block_max) {
      block_max = curr_max;
      block_id  = k;
    }
  }

  // revisit the block containing the max value
  block_start = block_id*block_size;
  p_x         = p_x-n+block_start;
  zmm_max     = _mm512_broadcastss_ps(_mm_broadcast_ss(&block_max));

  for (int j=0; j<block_size; j+=NUM_ELEMS_IN_REG) {
      zmm_x0 = _mm512_loadu_ps(p_x);
      m0 = _mm512_cmp_ps_mask(zmm_x0, zmm_max, _CMP_EQ_OQ);
      if (m0) {
        offset = _bit_scan_forward(m0);
        return (block_start + j + offset);
      }
      p_x += NUM_ELEMS_IN_REG;
  }

  return -1;
}
