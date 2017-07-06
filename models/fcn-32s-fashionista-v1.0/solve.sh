#!/bin/bash
#$-V
#$-cwd
#$-j y
#$-pe gpu 1
#$-l ga=gtx-titan*,gpu=1
#$-o log/
#$-N train-fcn
#
# qsub models/fcn-32s-fashionista-v1.0/solve.sh

cd models/fcn-32s-fashionista-v1.0/
export PYTHONPATH=$PYTHONPATH:../../python
python solve.py
cd ../fcn-16s-fashionista-v1.0/
python solve.py
cd ../fcn-8s-fashionista-v1.0/
python solve.py
echo "DONE"
