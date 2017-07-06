DATASETS=(
  # "tmm_dataset_sharing"
  # "fashionista-v1.0"
  "fashionista-v0.2"
  )
MODELS=(
  "attrlog"
  # "segf-8s"
  # "segc-8s-pre"
  )

for dataset in ${DATASETS[@]}; do
  for model in ${MODELS[@]}; do
	if [ ${dataset} = "tmm_dataset_sharing" ] 
	then
	  MODEL_SUFFIX="tmm"
	else
	  MODEL_SUFFIX=${dataset}
	fi

	python /ceph/tangseng/fashion-parsing/examples/fashionpose/plot_iteration.py ../models/${model}-${MODEL_SUFFIX}/train.log --start 0
	done
done