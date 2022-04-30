int ref_max (int *p_n, float *p_x)
{
  int n = *p_n;
  float max_val = FLT_MIN;
  int idx;

  for (int i=0; i<n; i++) {
    if (p_x[i] > max_val) {
      max_val = p_x[i];
      idx = i;
    }
  }
  return idx;
}
