# Learn an FCN model.
#
# python examples/fashionfcn/solve.py \
#     --model=models/fcn-32s-fashionista-v1.0 \
#     --base_weights=models/fcn-32s-pascalcontext/fcn-32s-pascalcontext.caffemodel \
#     --gpu=0
#

from __future__ import division
import argparse
import numpy as np
import os
import sys
sys.path.append('python')
import caffe

# Default base weights to fine-tune.
DEFAULT_BASE = 'models/fcn-32s-pascalcontext/fcn-32s-pascalcontext.caffemodel'
MODEL_DIR = 'models'


# make a bilinear interpolation kernel
# credit @longjon
def upsample_filt(size):
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)


# set parameters s.t. deconvolutional layers compute bilinear interpolation
# N.B. this is for deconvolution without groups
def interp_surgery(net, layers):
    for l in layers:
        m, k, h, w = net.params[l][0].data.shape
        if m != k:
            print 'input + output channels need to be the same'
            raise
        if h != w:
            print 'filters need to be square'
            raise
        filt = upsample_filt(h)
        net.params[l][0].data[range(m), range(k), :, :] = filt


# Set up and run the caffe solver.
def solve(model, base_weights, gpu):
    # init
    if gpu is not None:
        caffe.set_mode_gpu()
        caffe.set_device(int(gpu))
    else:
        caffe.set_mode_cpu()
    solver = caffe.SGDSolver(os.path.join(MODEL_DIR, model, 'solver.prototxt'))
    # do net surgery to set the deconvolution weights for bilinear interpolation
    interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
    interp_surgery(solver.net, interp_layers)
    # copy base weights for fine-tuning
    solver.net.copy_from(base_weights)
    # solve straight through -- a better approach is to define a solving loop to
    # 1. take SGD steps
    # 2. score the model by the test net `solver.test_nets[0]`
    # 3. repeat until satisfied
    solver.step(80000)
    del solver


# Parse input arguments.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Learn a FCN model.')
    parser.add_argument('--model', type=str, required=True,
                        help='Model name to fine-tune.')
    parser.add_argument('--base_weights', type=str, default=DEFAULT_BASE,
                        help='Default base weights to fine-tune.')
    parser.add_argument('--gpu', type=str, default=None,
                        help='GPU device id')
    args = parser.parse_args()
    solve(args.model, args.base_weights, args.gpu)
