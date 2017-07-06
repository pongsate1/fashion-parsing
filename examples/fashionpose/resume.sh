#!/bin/bash
#$-pe gpu 1
#$-l gpu=1,ga=gtx-titan*
#$-V
#$-j y
#$-o log/
#$-N train-fcn
#$-cwd

build/tools/caffe train \
  --solver=models/poseseg-32s-fashionista-v1.0/solver.prototxt \
  --snapshot=models/poseseg-32s-fashionista-v1.0/train_iter_170000.solverstate \
  --gpu=$SGE_GPU
