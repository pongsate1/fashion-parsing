#include "MaskRefine.h"

static float* UnaryPotential(const int* mask_in, float fg_prob, int label_num, int sample_num) {
  const float u_energy = -log( 1.0 / label_num );                  // Unknown sample
  const float n_energy = -log( (1.0 - fg_prob) / (label_num-1) );  // Negative sample
  const float p_energy = -log( fg_prob );                          // Positive sample

  float* unary = (float*)malloc(sample_num * label_num * sizeof(float));
  for( int k = 0, j = 0; k < sample_num; ++k, j+=label_num ){
    int index = mask_in[k];
    if (index >= label_num || index < 0) {
      for (int i = 0; i < label_num; ++i)
        unary[j+i] = u_energy;
    } else {
      for (int i = 0; i < label_num; ++i)
        unary[j+i] = n_energy;
      unary[j+index] = p_energy;
    }
  }

  return unary;
}



bool MaskRefine(const int* mask_in, const unsigned char* bgr_in,
  int width, int height, int label_num, const CRF_TAG* param, int* mask_out) {
  if (NULL == mask_in || NULL == mask_out || NULL == param || NULL == mask_out)
    return false;

  FullCRF crf(width, height, label_num);
  float* unary = UnaryPotential(mask_in, param->fg_prob_, label_num, width * height);
  crf.AddUnaryTerm(unary);
  crf.AddPairwiseBilateral(param->sigma_pos_bi_, param->sigma_pos_bi_, param->sigma_color_bi_,
    param->sigma_color_bi_, param->sigma_color_bi_, bgr_in, param->weight_bi_);
  crf.AddPairwisePosition(param->sigma_pos_, param->sigma_pos_, param->weight_pos_);
  //crf.AddPairwiseColor(param->sigma_color_, param->sigma_color_, param->sigma_color_, bgr_in, param->weight_color_);
  crf.Inference(param->iter_num, mask_out);

  if (unary) {
    free(unary);
    unary = NULL;
  }
  return true;
}