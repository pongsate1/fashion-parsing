#!/bin/bash
#
# Launch the FCN evaluator.
#
#   qsub examples/fashionfcn/eval.sh
#
#$-cwd
#$-V
#$-S /bin/bash
#$-o log/
#$-j y
#$-N eval-fcn
#-t 2
#
DATASETS=("fashionista-v0.2" "fashionista-v1.0")

SGE_TASK_ID=$((${SGE_TASK_ID=0}-1))
i=0

# Run evaluation.
for dataset in ${DATASETS[@]}; do
  if [ "$i" = "${SGE_TASK_ID}" ]; then
    python examples/fashionfcn/eval.sh \
      --inputs data/${dataset}/train-gt.lmdb \
               data/${dataset}/val-gt.lmdb \
               data/${dataset}/test-gt.lmdb \
      --predictions public/fashionfcn/fcn-32s-${dataset} \
                    public/fashionfcn/fcn-16s-${dataset} \
                    public/fashionfcn/fcn-8s-${dataset} \
      --labels data/${dataset}/labels.txt \
      --output public/fashionfcn/result-${dataset}.json
  fi
  i=`expr $i + 1`
done
