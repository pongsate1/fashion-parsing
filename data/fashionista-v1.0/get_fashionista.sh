#!/bin/bash
#
# Packaging
#
# tar czf caffe-fashionista-v1.0.tgz \
#   fashionista-v1.0/convert_fashionista_to_hdf5.m \
#   fashionista-v1.0/get_fashionista.sh
#
# SGE flags:
#
#$-V
#$-cwd
#$-j y
#$-N get_fashionista
#$-o log/
#$-S /bin/bash
#

#DIR="$( cd "$(dirname "$0")" ; pwd -P )"
DIR=data/fashionista-v1.0
cd $DIR

MATFILE="fashionista-v1.0.mat"

if [ -f $MATFILE ]
then
  echo "File exists: $MATFILE"
else
  echo "Downloading..."
  wget http://vision.is.tohoku.ac.jp/~kyamagu/private/fashionista2/data/fashionista-v1.0.mat
fi

echo "Creating HDF5 files..."

matlab -nodisplay <<MATLAB
run ../../startup.m
convert_fashionista_to_hdf5('Scale', 1);
MATLAB

echo "Done."
