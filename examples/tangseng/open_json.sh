MODEL='attrlog'
# MODEL='segc-8s-pre'
# MODEL='sege-8s'
DATASET='fashionista-v0.2'
# DATASET='fashionista-v1.0'
# DATASET='tmm'

DIR='public/fashionpose/'
FILENAME=${MODEL}-${DATASET}.json
FULLPATH=${DIR}${FILENAME}

python -m json.tool ${FULLPATH} | less
