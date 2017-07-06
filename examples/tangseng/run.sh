#!/bin/bash
#
# Launch the FCN test runner from different models and datasets.
#
#   qsub examples/fashionpose/run_main.sh
#
#$-cwd
#$-V
#$-S /bin/bash
#$-o run_prof_eval_attrlog_tmm.log
#$-j y
#$-l gpu=1,ga=gtx-titan*
#$-pe gpu 1
#$-N test_attrlog_tmm
#$-t 1
#

SCALE=1
DATASETS=(
  # "tmm_dataset_sharing"
  # "fashionista-v1.0"
  "fashionista-v0.2"
  )
MODELS=(
  "attrlog"
  # "sege-8s"
  )
ITERATION=120000
SNAPSHOT="train_iter_${ITERATION}.caffemodel"
MODEL_DIR="/ceph/tangseng/fashion-parsing/models/"
PUBLIC_DIR="/ceph/tangseng/fashion-parsing/public/fashionpose/"

SGE_TASK_ID=$((${SGE_TASK_ID=1}-1))
SGE_GPU=${SGE_GPU=0}
i=0

printf "HOSTNAME=%s\n" `hostname`

for dataset in ${DATASETS[@]}; do
  for model in ${MODELS[@]}; do
    if [ "${i}" = "${SGE_TASK_ID}" ]; then
      if [ ${dataset} = "tmm_dataset_sharing" ] 
      then
        MODEL_SUFFIX="tmm"
        input="/ceph/tangseng/fashion-parsing/data/${dataset}/TMM_test.h5"
      else
        MODEL_SUFFIX=${dataset}
        input="/ceph/tangseng/fashion-parsing/data/${dataset}/test-${SCALE}.h5"
      fi
      # input="data/${dataset}/train-${SCALE}.h5 \
      #        data/${dataset}/val-${SCALE}.h5 \
      #        data/${dataset}/test-${SCALE}.h5"
      python /ceph/tangseng/fashion-parsing/examples/fashionpose/run.py  \
        --model_def ${MODEL_DIR}${model}-${MODEL_SUFFIX}/deploy.prototxt \
        --model_weights ${MODEL_DIR}${model}-${MODEL_SUFFIX}/${SNAPSHOT} \
        --input ${input} \
        --output ${MODEL_DIR}${model}-${MODEL_SUFFIX}/test-iter${ITERATION}.h5 \
        --labels_clothing=/ceph/tangseng/fashion-parsing/data/${dataset}/labels.txt \
        --gpu=${SGE_GPU} \
        --batch_size=1

      cp ${MODEL_DIR}${model}-${MODEL_SUFFIX}/test-iter${ITERATION}.json ${PUBLIC_DIR}${model}-${MODEL_SUFFIX}.json
    fi
    i=`expr $i + 1`
  done
done
