#!/usr/bin/env sh
#$-pe gpu 1
#$-l gpu=1,gpu_arch=gtx-titan-x
#$-j y
#$-cwd
#$-V
#$-o ../models/segf-8s-fashionista-v1.0/resume.log

nohup python ../python/resume_train.py 2>&1 >> ../models/segf-8s-fashionista-v1.0/resume.log &
