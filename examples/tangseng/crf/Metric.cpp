#include "Metric.h"

float AllPixelAcc(const int* mask, const int* gt, int width, int height) {
  int pos = 0, neg = 0;
  for (int i = 0; i < width * height; ++i) {
    if (mask[i] == gt[i])
      ++pos;
    else
      ++neg;
  }
  return (float)pos / (neg + pos);
}
float FgdPixelAcc(const int* mask, const int* gt, int width, int height, int bgd_id) {
  int pos = 0, neg = 0;
  for (int i = 0; i < width * height; ++i) {
    if (bgd_id != gt[i]) {
      if (mask[i] == gt[i])
        ++pos;
      else
        ++neg;
    }
  }
  if (0 == neg + pos)
    return 1.f;
  return (float)pos / (neg + pos);
}