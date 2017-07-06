#pragma once
/* Full-connected CRF.
 * Original paper: "Efficient Inference in Fully Connected CRFs with Gaussian Edge Potentials", NIPS 2011
 * Original code: http://graphics.stanford.edu/projects/densecrf/
 * WU ZHIPENG
 * 2015/12/08
 * version 1.0
 */
#include <cmath>
#include <vector>
#include "permutohedral.h"

using namespace std;

namespace full_crf {

class PairwisePotential {
public:
  PairwisePotential(const float* features, int D, int N, float w);
  ~PairwisePotential();
  /* Member functions */
  void apply(float* out_values, const float* in_values, float* tmp, int value_size) const;
private:
  /* Disable Copyable and Movable Initialization */
  PairwisePotential(PairwisePotential& other);
  PairwisePotential(const PairwisePotential& other);
  PairwisePotential& operator=(PairwisePotential&& other);
  PairwisePotential& operator=(const PairwisePotential& other);

  /* Member variables */
  Permutohedral lattice_;
  int D_;                 // dimension.
  int N_;                 // number of vertexes.
  float W_;               // weighting.
  float* norm_;
};

class FullCRF {
public:
  FullCRF(int W, int H, int M);
  ~FullCRF();
  /* Member functions */
  void AddUnaryTerm(const float* unary);
  void AddPairwisePosition(float sx, float sy, float w);
  void AddPairwiseColor(float sr, float sg, float sb, const unsigned char *im, float w);
  void AddPairwiseBilateral(float sx, float sy, float sr, float sg, float sb, const unsigned char * im, float w);
  void Inference(int iter_num, int* result, float relax = 1.0);

private:
  /* Disable Copyable and Movable Initialization */
  FullCRF(FullCRF& other);
  FullCRF(const FullCRF& other);
  FullCRF& operator=(FullCRF&& other);
  FullCRF& operator=(const FullCRF& other);

  void AddPairwiseEnergy(const float * features, int D, float w = 1.0f);
  void ExpAndNormalize(float* out, const float* in, float scale = 1.0, float relax = 1.0);

  /* Member variables */
  int M_;                 // number of labels.
  int W_;                 // image width.
  int H_;                 // image height.
  int N_;                 // vertexes number.
  float* unary_;          // saving unary term.
  float* current_;        // saving pairwise term (current round).
  float* next_;           // saving pairwise term (next round).
  float* tmp_;            // auxiliary memory.
  vector<PairwisePotential*> pairwise_;
};

}   // full_crf;