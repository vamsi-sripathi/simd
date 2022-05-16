int max_idx_tracking (int *p_n, float *p_x) 
{
  __m512 zmm_x0, zmm_x1, zmm_x2, zmm_x3, zmm_x4, zmm_x5, zmm_x6, zmm_x7;
  __m512 zmm_curr_max, zmm_tmp0, zmm_tmp1, zmm_tmp2, zmm_tmp3, zmm_tmp4, zmm_tmp5;
  __m256 ymm_x0, ymm_x1;
  __m128 xmm_x0, xmm_x1;
  __mmask16 m0;
  __m512 zmm_max      = _mm512_set1_ps(FLT_MIN);
  __m512i zmm_indices = _mm512_setzero_epi32();

  int n = *p_n;
  int offset, block_idx;
  float max_val;

  for (int j=0; j<n; j+=N_UNROLL) {
    zmm_x0   = _mm512_loadu_ps(p_x);
    zmm_x1   = _mm512_loadu_ps(p_x+16);
    zmm_x2   = _mm512_loadu_ps(p_x+32);
    zmm_x3   = _mm512_loadu_ps(p_x+48);
    zmm_tmp0 = _mm512_max_ps(zmm_x0, zmm_x1);
    zmm_tmp1 = _mm512_max_ps(zmm_x2, zmm_x3);
#if N_UNROLL > 64
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

    m0 = _mm512_cmp_ps_mask(zmm_curr_max, zmm_max, _CMP_GT_OS);
    if (1/*m0*/) {
      zmm_max      = _mm512_mask_blend_ps(m0, zmm_max, zmm_curr_max);
      zmm_indices  = _mm512_mask_set1_epi32(zmm_indices, m0, j);
    }
    p_x += N_UNROLL;
  }

  p_x = p_x-n;

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
  _mm_store_ss(&max_val, xmm_x0);


  m0 = _mm512_cmp_ps_mask(_mm512_broadcastss_ps(xmm_x0), zmm_max, _CMP_EQ_OQ);
  unsigned int pop_cnt = _mm_popcnt_u32(_cvtmask16_u32(m0));
  if (pop_cnt == 1) {
    _mm512_mask_compressstoreu_epi32(&block_idx, m0, zmm_indices);
    offset = _bit_scan_forward(m0);
  } else {
    __m256i ymm_i0, ymm_i1;
    __m128i xmm_i0, xmm_i1;

    zmm_indices = _mm512_mask_blend_epi32(m0, _mm512_set1_epi32(INT_MAX), zmm_indices);
    // reduction tree on zmm register to find min. index value
    ymm_i0 = _mm512_castsi512_si256(zmm_indices);
    ymm_i1 = _mm512_extracti32x8_epi32(zmm_indices, 1);
    ymm_i0 = _mm256_min_epi32(ymm_i0, ymm_i1);

    xmm_i0 = _mm256_castsi256_si128(ymm_i0);
    xmm_i1 = _mm256_extracti32x4_epi32(ymm_i0, 1);
    xmm_i0 = _mm_min_epi32(xmm_i0, xmm_i1);

    xmm_i1 = _mm_shuffle_epi32(xmm_i0, 0x0E);
    xmm_i0 = _mm_min_epi32(xmm_i0, xmm_i1);

    xmm_i1 = _mm_shuffle_epi32(xmm_i0, 0x01);
    xmm_i0 = _mm_min_epi32(xmm_i0, xmm_i1);
    _mm_store_ss((float *)&block_idx, _mm_castsi128_ps(xmm_i0));

    __mmask16 m1 = _mm512_cmp_epi32_mask(
                   _mm512_broadcastd_epi32(xmm_i0), zmm_indices, _MM_CMPINT_EQ);
    if (_mm_popcnt_u32(_cvtmask16_u32(m1)) > 1) {
      p_x += block_idx;
      zmm_max = _mm512_broadcastss_ps(xmm_x0);

      for (int j=0; j<N_UNROLL; j+=NUM_ELEMS_IN_REG) {
        zmm_x0 = _mm512_loadu_ps(p_x);
        m0     = _mm512_cmp_ps_mask(zmm_x0, zmm_max, _CMP_EQ_OQ);
        if (m0) {
          return (block_idx + j + _bit_scan_forward(m0));
        }
        p_x += NUM_ELEMS_IN_REG;
      }
    }
    offset = _bit_scan_forward(m1);
  }

  for (int i=0; i<(N_UNROLL/NUM_ELEMS_IN_REG); i++) {
    if (max_val == p_x[block_idx+offset+(i*NUM_ELEMS_IN_REG)]) {
      return block_idx + offset + (i*NUM_ELEMS_IN_REG);
    }
  }

  return -1;
}
