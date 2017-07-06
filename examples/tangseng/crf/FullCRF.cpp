#include "FullCRF.h"

#ifndef SAFE_FREE(buf)
#define SAFE_FREE(buf)               {if( NULL!=(buf)  ) {free(buf); (buf)=NULL; } }
#endif

namespace full_crf {

PairwisePotential::PairwisePotential(const float* features, int D, int N, float W) : D_(D), N_(N), W_(W) {
  /* Compute & save normalization factor. */
  lattice_.init(features, D_, N_);              // TODO: parallel this function.
  norm_ = (float*)malloc(N_ * sizeof(float));
  for (int i = 0; i < N_; ++i)
    norm_[i] = 1.f;
  lattice_.compute(norm_, norm_, 1);            // TODO: parallel this function.
  for (int i = 0; i < N_; ++i)
    norm_[i] = W_ / (norm_[i]+1e-20f);
}
PairwisePotential::~PairwisePotential() {
  SAFE_FREE(norm_);
}
void PairwisePotential::apply(float* out_values, const float* in_values, float* tmp, int value_size) const {
  lattice_.compute(tmp, in_values, value_size);   // TODO: parallel this function.
  for ( int i=0,k=0; i<N_; i++ )
    for ( int j=0; j<value_size; j++, k++ )
      out_values[k] += norm_[i]*tmp[k];
}

FullCRF::FullCRF(int W, int H, int M ) : M_(M), W_(W), H_(H), N_(W * H) {
  unary_   = (float*)malloc(N_ * M_ * sizeof(float));
  current_ = (float*)malloc(N_ * M_ * sizeof(float));
  next_    = (float*)malloc(N_ * M_ * sizeof(float));
  tmp_     = (float*)malloc(N_ * M_ * sizeof(float) * 2);
}
FullCRF::~FullCRF() {
  SAFE_FREE(unary_);
  SAFE_FREE(current_);
  SAFE_FREE(next_);
  SAFE_FREE(tmp_);
  for (size_t i = 0; i < pairwise_.size(); ++i)
    delete pairwise_[i];
}
void FullCRF::AddPairwiseEnergy (const float* features, int D, float w) {
  pairwise_.push_back(new PairwisePotential(features, D, N_, w));
}
void FullCRF::ExpAndNormalize (float* out, const float* in, float scale, float relax) {
  const float * b = in;
  float* a = out;
  float mx, tt;
  float *V = (float*)malloc((N_+10) * sizeof(float));
  for(int i=0; i<N_; i++, b+=M_, a+=M_){
    // Find the max and subtract it so that the exp doesn't explode
    mx = scale*b[0];
    for(int j=0; j<M_; j++) {
      V[j] = scale*b[j];
      if( mx < V[j] )
        mx = V[j];
    }
    tt = 0;
    for(int j=0; j<M_; j++){
      V[j] = exp(V[j]-mx);
      tt += V[j];
    }
    // Make it a probability
    for(int j=0; j<M_; j++)
      V[j] /= tt;

    for(int j=0; j<M_; j++)
      if (relax == 1)
        a[j] = V[j];
      else
        a[j] = (1-relax)*a[j] + relax*V[j];
  }
  SAFE_FREE(V);
}
void FullCRF::AddUnaryTerm(const float* unary) {
  memcpy(unary_, unary, N_ * M_ * sizeof(float));
}
void FullCRF::AddPairwisePosition (float sx, float sy, float w) {
  float * feature = (float*)malloc(N_ * 2 * sizeof(float));
  for(int j=0; j<H_; j++)
    for(int i=0; i<W_; i++) {
      feature[(j*W_+i)*2+0] = i / sx;
      feature[(j*W_+i)*2+1] = j / sy;
    }
    AddPairwiseEnergy(feature, 2, w);
    SAFE_FREE(feature);
}
void FullCRF::AddPairwiseColor(float sr, float sg, float sb, const unsigned char *im, float w) {
  float *feature = (float*)malloc(N_ * 3 * sizeof(float));
  for( int j=0; j<H_; j++)
    for( int i=0; i<W_; i++) {
      feature[(j*W_+i)*3+0] = im[(i+j*W_)*3+0] / sr;
      feature[(j*W_+i)*3+1] = im[(i+j*W_)*3+1] / sg;
      feature[(j*W_+i)*3+2] = im[(i+j*W_)*3+2] / sb;
    }
    AddPairwiseEnergy(feature, 3, w);
    SAFE_FREE(feature);
}
void FullCRF::AddPairwiseBilateral(float sx, float sy, float sr, float sg, float sb, const unsigned char * im, float w) {
  float * feature = (float*)malloc(N_ * 5 * sizeof(float));
  for(int j=0; j<H_; j++)
    for(int i=0; i<W_; i++) {
      feature[(j*W_+i)*5+0] = i / sx;
      feature[(j*W_+i)*5+1] = j / sy;
      feature[(j*W_+i)*5+2] = im[(i+j*W_)*3+0] / sr;
      feature[(j*W_+i)*5+3] = im[(i+j*W_)*3+1] / sg;
      feature[(j*W_+i)*5+4] = im[(i+j*W_)*3+2] / sb;
    }
    AddPairwiseEnergy(feature, 5, w);
    SAFE_FREE(feature);
}
void FullCRF::Inference(int iter_num, int* result, float relax) {
  /* Initialize using the unary energies. */
  ExpAndNormalize(current_, unary_, -1);               // TODO: parallel this function.

  /* Do inference round by round. */
  for(int it = 0; it < iter_num; it++) {
    /* Add unary term to next_. */
    for(int i = 0; i < N_ * M_; i++)
      next_[i] = -unary_[i];
    /* Add pairwise term to next_. */
    for(size_t i = 0; i < pairwise_.size(); i++)
      pairwise_[i]->apply(next_, current_, tmp_, M_);
    /* Calculate prob & normalize. */
    ExpAndNormalize(current_, next_, 1.0, relax);      // TODO: parallel this function.
  }

  /* Return final label based on MAP. */
  const float* p = current_;
  float mx;
  int ind;
  for (int i = 0; i < N_; i++, p+=M_) {
    ind = 0;
    mx = p[0];
    for (int j = 1; j < M_; j++) {
      if (mx < p[j]) {
        mx = p[j];
        ind = j;
      }
    }
    result[i] = ind;
  }
}

} // namespace full_crf;