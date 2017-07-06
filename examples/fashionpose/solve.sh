#!/bin/bash
#
# Launch caffe training process using SGE.
#
#$-V
#$-cwd
#$-j y
#$-pe gpu 1
#$-l ga=gtx-titan-x,gpu=1
#$-o log/
#$-S /bin/bash
#$-N fashion-fcn-train
#$-t 1
#
# Usage:
#
# qsub examples/fashionpose/solve.sh
#

DATASETS=(
  "fashionista-v1.0"
  "fashionista-v0.2"
  )
SGE_TASK_ID=$((${SGE_TASK_ID=1}-1))
SGE_GPU=$(echo ${SGE_GPU=0})
SCALE=1
MAX_ITER=100000

hostname

for j in ${!DATASETS[@]}; do
  if [ "${j}" = "${SGE_TASK_ID}" ]; then
    # Set up input paths.
    dataset=${DATASETS[$j]}
    models=(
      "pose-16s1-${dataset}"
      "segc-8s-pre-${dataset}"
      "segf-8s-${dataset}"
      "sege-8s-${dataset}"
      "segd-8s-${dataset}"
      "segc-8s-${dataset}"
      "segb-8s-${dataset}"
      "sega-8s-${dataset}"
      "sega-16s-${dataset}"
      "poseseg-8s-${dataset}"
      "sega-32s-${dataset}"
      "pose-8s-${dataset}"
      "pose-16s-${dataset}"
      )
    base_weights=(
      "models/pose-16s-${dataset}/train_iter_${MAX_ITER}.caffemodel"
      "models/fcn-8s-${dataset}/train_iter_80000.caffemodel"
      "models/segc-8s-pre-${dataset}/train1000_iter_100000.caffemodel"
      "models/segc-8s-pre-${dataset}/train1000_iter_100000.caffemodel"
      "models/segc-8s-pre-${dataset}/train1000_iter_100000.caffemodel"
      "models/segc-8s-pre-${dataset}/train1000_iter_90000.caffemodel"
      "models/fcn-8s-${dataset}/train_iter_80000.caffemodel"
      "models/fcn-8s-${dataset}/train_iter_80000.caffemodel"
      "models/fcn-8s-${dataset}/train_iter_80000.caffemodel"
      "models/sega-32s-${dataset}/train_iter_100000.caffemodel"
      "models/poseseg-16s-${dataset}/train_iter_${MAX_ITER}.caffemodel"
      "models/fcn-32s-${dataset}/train_iter_80000.caffemodel"
      "models/pose-16s-${dataset}/train_iter_${MAX_ITER}.caffemodel"
      "models/pose-32s-${dataset}/train_iter_${MAX_ITER}.caffemodel"
      )
    #base_weights=(
    #  "models/fcn-32s-pascalcontext/fcn-32s-pascalcontext.caffemodel"
    #  )
    # Run the job sequentially.
    #for i in ${!models[@]}; do
    for i in 0; do
      echo "# Training ${models[$i]} using GPU=${SGE_GPU}"
      python examples/fashionpose/solve.py \
        --model=${models[$i]} \
        --base_weights=${base_weights[$i]} \
        --max_iter=${MAX_ITER} \
        --gpu=${SGE_GPU} 2>&1 | \
        tee models/${models[$i]}/train.log
    done
    echo "DONE"
  fi
done
