DATASETS=(
  "tmm_dataset_sharing"
  "fashionista-v1.0"
  "fashionista-v0.2"
  )
MODELS=(
  "attrlog"
  # "sege-8s"
  )

MODEL_DIR="models/"
PUBLIC_DIR="public/fashionpose/"

printf "HOSTNAME=%s\n" `hostname`

for dataset in ${DATASETS[@]}; do
  for model in ${MODELS[@]}; do
    if [ ${dataset} = "tmm_dataset_sharing" ] 
    then
      MODEL_SUFFIX="tmm"
    else
      MODEL_SUFFIX=${dataset}
    fi
  
    python examples/tangseng/crf_email_eval.py  \
      --input ${MODEL_DIR}${model}-${MODEL_SUFFIX}/refine/ \
      --output ${MODEL_DIR}${model}-${MODEL_SUFFIX}/refine/output.h5 \
      --gt data/${dataset}/testimages/ \
      --labels_clothing=data/${dataset}/labels.txt

    cp ${MODEL_DIR}${model}-${MODEL_SUFFIX}/refine/output.json ${PUBLIC_DIR}${model}-${MODEL_SUFFIX}-crf.json
  done
done
