DATASETS=(
  "tmm_dataset_sharing"
  "fashionista-v1.0"
  "fashionista-v0.2"
  )
MODELS=(
  "attrlog"
  "sege-8s"
  )

for dataset in ${DATASETS[@]}; do
  for model in ${MODELS[@]}; do
    ./examples/tangseng/crf_smoothing ${model} ${dataset}
  done
done
