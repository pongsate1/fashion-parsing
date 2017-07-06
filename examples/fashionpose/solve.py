# Learn an FCN model.
#
# python examples/fashionfcn/solve.py \
#     --model=models/poseseg-32s-fashionista-v1.0 \
#     --base_weights=models/fcn-32s-pascalcontext/fcn-32s-pascalcontext.caffemodel \
#     --gpu=0 2>&1 > /dev/null | \
#     tee models/poseseg-32s-fashionista-v1.0/train.log
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


# set parameters s.t. layers compute bilinear interpolation
# N.B. this is for deconvolution without groups
def copy_fork_weights(net, pose_layers):
    for pose_layer in pose_layers:
        base_layer = pose_layer.replace('pose_', '')
        for i in xrange(len(net.params[pose_layer])):
            net.params[pose_layer][i].data[...] = net.params[base_layer][i].data


# Set up and run the caffe solver.
def solve(model, base_weights, gpu, max_iter):
    # init
    if gpu is not None:
        caffe.set_mode_gpu()
        caffe.set_device(int(gpu))
        print("Using GPU={0}".format(int(gpu)))
    else:
        caffe.set_mode_cpu()
    solver = caffe.SGDSolver(os.path.join(MODEL_DIR, model, 'solver.prototxt'))
    # do net surgery to set the deconvolution weights for bilinear interpolation
    interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
    interp_surgery(solver.net, interp_layers)
    # copy base weights for fine-tuning
    solver.net.copy_from(base_weights)
    # Hack to copy pre-trained weights between forks
    if 'pascalcontext' in base_weights and 'fork' in model:
        print("Copying fc layers")
        pose_fc_layers = [k for k in solver.net.params.keys() if 'pose_fc' in k]
        copy_fork_weights(solver.net, pose_fc_layers)
    # solve straight through -- a better approach is to define a solving loop to
    # 1. take SGD steps
    # 2. score the model by the test net `solver.test_nets[0]`
    # 3. repeat until satisfied
    solver.step(max_iter)
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
    parser.add_argument('--max_iter', type=int, default=100000,
                        help='Maximum iterations.')
    args = parser.parse_args()
    solve(args.model, args.base_weights, args.gpu, args.max_iter)
