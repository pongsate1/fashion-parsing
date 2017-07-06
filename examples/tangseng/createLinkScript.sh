DATASETS=(
  "tmm_dataset_sharing"
  "fashionista-v1.0"
  "fashionista-v0.2"
  )
MODELS=(
  "attrlog"
  "segf-8s"
  "sege-8s"
  )

MODEL_DIR="../../models/"
PUBLIC_DIR="public/fashionpose/"
for dataset in ${DATASETS[@]}; do
  for model in ${MODELS[@]}; do
    
    if [ ${dataset} = "tmm_dataset_sharing" ] 
    then
      MODEL_SUFFIX="tmm"
    else
      MODEL_SUFFIX=${dataset}
    fi
      
  	ln -s ${MODEL_DIR}${model}-${MODEL_SUFFIX}/mask ${PUBLIC_DIR}${model}-${MODEL_SUFFIX}
  	ln -s ${MODEL_DIR}${model}-${MODEL_SUFFIX}/refine ${PUBLIC_DIR}${model}-${MODEL_SUFFIX}-crf

  done
done
