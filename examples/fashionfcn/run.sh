#!/bin/bash
#
# Launch the FCN test runner from different models and datasets.
#
#   qsub examples/fashionfcn/run_main.sh
#
#$-cwd
#$-V
#$-S /bin/bash
#$-o log/
#$-j y
#$-l gpu=1,ga=gtx-titan*
#$-pe gpu 1
#$-N run-fcn
#$-t 6
#

DATASETS=("fashionista-v0.2" "fashionista-v1.0")
MODELS=("fcn-32s" "fcn-16s" "fcn-8s")
SNAPSHOT="train_iter_10000.caffemodel"

SGE_TASK_ID=$((${SGE_TASK_ID=0}-1))
SGE_GPU=${SGE_GPU=0}
i=0

for dataset in ${DATASETS[@]}; do
  for model in ${MODELS[@]}; do
    if [ "${i}" = "${SGE_TASK_ID}" ]; then
      input="data/${dataset}/train.lmdb \
             data/${dataset}/val.lmdb \
             data/${dataset}/test.lmdb"
      python examples/fashionfcn/run.py  \
        --model_def models/${model}-${dataset}/deploy.prototxt \
        --model_weights models/${model}-${dataset}/${SNAPSHOT} \
        --input ${input} \
        --output public/fashionfcn/${model}-${dataset}/ \
        --gpu=${SGE_GPU}
    fi
    i=`expr $i + 1`
  done
done
