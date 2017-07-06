#!/bin/bash
#
# Main experimental script for Fashion FCN.
#
#   ./examples/fashionfcn/main.sh
#

qsub -sync examples/fashionfcn/solve.sh

qsub -sync examples/fashionfcn/run.sh

qsub -sync examples/fashionfcn/eval.sh

qsub -sync examples/fashionfcn/export.sh

qsub -sync examples/fashionfcn/package.sh
