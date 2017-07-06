#!/bin/bash
#
# Launch caffe training process using SGE.
#
#$-V
#$-cwd
#$-j y
#$-pe gpu 1
#$-l ga=gtx-titan*,gpu=1
#$-o log/
#$-S /bin/bash
#$-N train-fcn
#$-t 2
#
# Usage:
#
# qsub examples/fashionfcn/solve.sh
#

DATASETS=("fashionista-v0.2" "fashionista-v1.0")
SGE_TASK_ID=$((${SGE_TASK_ID=0}-1))
SGE_GPU=${SGE_GPU=0}

for j in ${!DATASETS[@]}; do
  if [ "${j}" = "${SGE_TASK_ID}" ]; then
    # Set up input paths.
    dataset=${DATASETS[$j]}
    models=(
      "fcn-32s-${dataset}"
      "fcn-16s-${dataset}"
      "fcn-8s-${dataset}"
      )
    base_weights=(
      "models/fcn-32s-pascalcontext/fcn-32s-pascalcontext.caffemodel"
      "models/fcn-32s-${dataset}/train_iter_80000.caffemodel"
      "models/fcn-16s-${dataset}/train_iter_80000.caffemodel"
      )
    # Run the job sequentially.
    for i in ${!models[@]}; do
      echo "# Training ${models[$i]} using GPU=${SGE_GPU}"
      python examples/fashionfcn/solve.py \
        --model=${models[$i]} \
        --base_weights=${base_weights[$i]}/ \
        --gpu=${SGE_GPU}
    done
    echo "DONE"
  fi
done
