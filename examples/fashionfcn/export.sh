#!/bin/bash
#
# Export the dataset into the result format.
#
#   qsub examples/fashionfcn/export.sh
#
#$-cwd
#$-V
#$-S /bin/bash
#$-o log/
#$-j y
#$-N export-fcn
#-t 3
#
DATASETS=("fashionista-v0.2" "fashionista-v1.0")

SGE_TASK_ID=$((${SGE_TASK_ID=0}-1))
i=0

if [ "${i}" = "${SGE_TASK_ID}" ]; then
  # Make the index file.
  python ./examples/fashionfcn/export_index.py \
    --datasets $(printf " data/%s" ${DATASETS[@]}) \
    --splits train val test \
    --labels labels.txt \
    --output public/fashionfcn/metadata.json

  # Images are the same.
  dataset=${DATASETS[0]}
  images="data/${dataset}/train.lmdb \
          data/${dataset}/val.lmdb \
          data/${dataset}/test.lmdb"
  python examples/fashionfcn/export.py \
    --input ${images} \
    --output public/fashionfcn/images/ \
    --format jpg
fi

# Annotations are different.
for dataset in ${DATASETS[@]}; do
  i=`expr $i + 1`
  if [ "${i}" = "${SGE_TASK_ID}" ]; then
    labels="data/${dataset}/train-gt.lmdb \
            data/${dataset}/val-gt.lmdb \
            data/${dataset}/test-gt.lmdb"
    python examples/fashionfcn/export.py \
      --input ${labels} \
      --output public/fashionfcn/truth-${dataset}/ \
      --format png
  fi
done
