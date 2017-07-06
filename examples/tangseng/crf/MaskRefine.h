#pragma once
#include "FullCRF.h"
using namespace full_crf;

struct CRF_TAG {
  /* Unary potential param. */
  float fg_prob_;

  /* PairwisePosition potential param. */
  float sigma_pos_;
  float weight_pos_;

  /* PairwiseColor potential param. */
  float sigma_color_;
  float weight_color_;

  /* PairwiseBilateral potential param. */
  float sigma_pos_bi_;
  float sigma_color_bi_;
  float weight_bi_;

  /* Inference param. */
  int iter_num;
};

bool MaskRefine(const int* mask_in, const unsigned char* bgr_in,
  int width, int height, int label_num, const CRF_TAG* param, int* mask_out);