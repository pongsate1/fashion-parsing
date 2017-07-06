#!/bin/bash
#
# Launch the FCN test runner from different models and datasets.
#
#   qsub examples/fashionpose/run_main.sh
#
#$-cwd
#$-V
#$-S /bin/bash
#$-o log/
#$-j y
#$-l gpu=1,ga=gtx-titan*
#$-pe gpu 1
#$-N run-fcn
#$-t 1
#

SCALE=1
DATASETS=(
  "fashionista-v1.0"
  # "fashionista-v0.2"
  )
MODELS=(
  # "segf-8s"
  # "sege-8s"

  # "segd-8s"
  # "fcn-32s"
  # "fcn-16s"
  "fcn-8s"
  # "segc-8s"
  # "pose-8s"
  # "poseseg-8s"
  # "sega-8s"
  # "segb-8s"
  # "sega-16s"
  # "poseseg-16s"
  # "sega-32s"
  # "fork5-32s"
  # "poseseg-32s"
  # "pose-16s"
  )
ITERATION=80000
SNAPSHOT="train_iter_${ITERATION}.caffemodel"

SGE_TASK_ID=$((${SGE_TASK_ID=1}-1))
SGE_GPU=${SGE_GPU=0}
i=0

printf "HOSTNAME=%s\n" `hostname`

for dataset in ${DATASETS[@]}; do
  for model in ${MODELS[@]}; do
    if [ "${i}" = "${SGE_TASK_ID}" ]; then
      # input="data/${dataset}/train-${SCALE}.h5 \
      #        data/${dataset}/val-${SCALE}.h5 \
      #        data/${dataset}/test-${SCALE}.h5"
      input="data/${dataset}/test-${SCALE}.h5"
      python examples/fashionpose/run.py  \
        --model_def models/${model}-${dataset}/deploy.prototxt \
        --model_weights models/${model}-${dataset}/${SNAPSHOT} \
        --input ${input} \
        --output public/fashionpose/${model}-${dataset}-test-iter${ITERATION}.h5 \
        --labels_clothing=data/${dataset}/labels.txt \
        --labels_joint=data/${dataset}/joints.txt \
        --gpu=${SGE_GPU} \
        --batch_size=2
    fi
    i=`expr $i + 1`
  done
done
