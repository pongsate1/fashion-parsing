#!/bin/bash
#
# Package the datasets for distribution.
#
#   ./examples/fashionfcn/package.sh
#
#$-cwd
#$-V
#$-S /bin/bash
#$-o log/
#$-j y
#$-N package
#

VERSION=`date +"%Y%m%d"`

# Copy information.
cp examples/fashionfcn/README.md public/fashionfcn/

# GZIP data.
cd public/
tar --exclude='*.tgz' -czvf \
  fashionfcn-${VERSION}.tgz \
  fashionfcn/
echo "DONE"
