DATASETS=(
  "tmm_dataset_sharing"
  "fashionista-v1.0"
  "fashionista-v0.2"
  )
MODELS=(
  "attrlog"
  # "sege-8s"
  )
ITERATION=120000
SNAPSHOT="train_iter_${ITERATION}.caffemodel"
MODEL_DIR="models/"
PUBLIC_DIR="public/fashionpose/"

SGE_GPU=${SGE_GPU=0}

printf "HOSTNAME=%s\n" `hostname`

for dataset in ${DATASETS[@]}; do
  for model in ${MODELS[@]}; do
    
      if [ ${dataset} = "tmm_dataset_sharing" ] 
      then
        MODEL_SUFFIX="tmm"
        input="data/${dataset}/TMM_test.h5"
      else
        MODEL_SUFFIX=${dataset}
        input="data/${dataset}/test-${SCALE}.h5"
      fi
      
      python examples/fashionpose/run.py  \
        --model_def ${MODEL_DIR}${model}-${MODEL_SUFFIX}/deploy.prototxt \
        --model_weights ${MODEL_DIR}${model}-${MODEL_SUFFIX}/${SNAPSHOT} \
        --input ${input} \
        --output ${MODEL_DIR}${model}-${MODEL_SUFFIX}/test-iter${ITERATION}.h5 \
        --labels_clothing=data/${dataset}/labels.txt \
        --gpu=${SGE_GPU} \
        --batch_size=1

      cp ${MODEL_DIR}${model}-${MODEL_SUFFIX}/test-iter${ITERATION}.json ${PUBLIC_DIR}${model}-${MODEL_SUFFIX}.json
  
  done
done
