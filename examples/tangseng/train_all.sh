# Launch the FCN test runner for different models and datasets.

DATASETS=(
  "tmm_dataset_sharing"
  "fashionista-v1.0"
  "fashionista-v0.2"
  )
MODELS=(
  "fcn-32s"
  "fcn-16s"
  "fcn-8s"
  "segc-8s-pre"   #Attribute layers training
  "sege-8s"       #Attribute broadcasting
  "attrlog"       #Attribute filtering
  )
base_model="fcn-32s"
ITERATION=80000
SNAPSHOT="train_iter_${ITERATION}.caffemodel"
MODEL_DIR="models/"
PUBLIC_DIR="public/fashionpose/"
SCRIPT_DIR="examples/tangseng/"

SGE_GPU=${SGE_GPU=0}

printf "HOSTNAME=%s\n" `hostname`

for dataset in ${DATASETS[@]}; do
  if [ ${dataset} = "tmm_dataset_sharing" ] 
  then
    MODEL_SUFFIX="tmm"
    input="data/${dataset}/TMM_test.h5"
  else
    MODEL_SUFFIX=${dataset}
  fi
  for model in ${MODELS[@]}; do    
    if [ ${model} = "fcn-32s" ] # finetune from fcn-32s-pascalcontext
    then
      python ${SCRIPT_DIR}train.py  \
        --solver ${MODEL_DIR}${model}-${MODEL_SUFFIX}/solver.prototxt \
        --base_weights ${MODEL_DIR}${base_model}-pascalcontext/fcn-32s-pascalcontext.caffemodel \
        --out_model ${MODEL_DIR}${model}-${MODEL_SUFFIX}/train_iter_${ITERATION}.caffemodel \
        --iter ${ITERATION} \
        2>&1 >>/dev/stdout #> ${MODEL_DIR}${model}-${MODEL_SUFFIX}/train_draft.log &
    else                        # finetune from parent model in the hierarchy
      python ${SCRIPT_DIR}train.py \
        --solver ${MODEL_DIR}${model}-${MODEL_SUFFIX}/solver.prototxt \
        --base_weights ${MODEL_DIR}${base_model}-${dataset}/train_iter_80000.caffemodel \
        --out_model ${MODEL_DIR}${model}-${MODEL_SUFFIX}/train_iter_${ITERATION}.caffemodel \
        --iter ${ITERATION} \
        2>&1 >>/dev/stdout #> ${MODEL_DIR}${model}-${MODEL_SUFFIX}/train_draft.log &
    fi
        
    if [ ${base_model} != "segc-8s-pre" ]
    then
      base_model=${model}
    fi
  done
done