# Convert data in h5 to appropiate format for CRF

DATASETS=(
  "tmm_dataset_sharing"
  "fashionista-v1.0"
  "fashionista-v0.2"
  )
MODELS=(
  "attrlog"
  "sege-8s"
  )
ITERATION=80000
MODEL_DIR="models/"

for dataset in ${DATASETS[@]}; do
  
  if [ ${dataset} = "tmm_dataset_sharing" ]
  then
    python examples/fashionpose/export.py \
      --inputs data/tmm_dataset_sharing/TMM_test.h5 \
      --output data/tmm_dataset_sharing/testimages
  else
    python examples/fashionpose/export.py \
      --inputs data/${dataset}/test-1.h5 \
      --output data/${dataset}/testimages
  fi  

  for model in ${MODELS[@]}; do
    if [ ${dataset} = "tmm_dataset_sharing" ] 
    then
      MODEL_SUFFIX="tmm"
    else
      MODEL_SUFFIX=${dataset}
    fi

  	python examples/fashionpose/export.py \
  	--inputs ${MODEL_DIR}${model}-${MODEL_SUFFIX}/test-iter${ITERATION}.h5 \
  	--output ${MODEL_DIR}${model}-${MODEL_SUFFIX}/mask      
 	
  done
done