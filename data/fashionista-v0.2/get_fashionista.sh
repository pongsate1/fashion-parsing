#!/bin/bash
#
# Packaging
#
# tar czf caffe-fashionista_v0.2.tgz \
#   fashionista-v0.2/caffe_proto_.cc \
#   fashionista-v0.2/convert_fashionista_to_lmdb.m \
#   fashionista-v0.2/get_fashionista.sh
#
# Conversion
#
# echo run convert_fashionista_to_lmdb | matlab -nodisplay
#

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
cd $DIR

echo "Downloading..."

wget http://vision.is.tohoku.ac.jp/~kyamagu/research/paperdoll/fashionista-v0.2.1.tgz

echo "Unzipping..."

tar -xzf fashionista-v0.2.1.tgz && rm -rf fashionista-v0.2.1.tgz

echo "Done."

matlab -nodisplay <<MATLAB
run ../../startup.m
convert_fashionista_to_hdf5
MATLAB

echo "Done."
