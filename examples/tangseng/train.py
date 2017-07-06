# this file attempt to finetune a model for a dataset

# import stuff as usual

caffe_root = '/ceph/tangseng/fashion-parsing/'
import sys
sys.path.append(caffe_root + 'python')

import argparse
import os
import cPickle
import logging
import numpy as np
import pandas as pd
from PIL import Image as PILImage
#import Image
import cStringIO as StringIO
import caffe
import matplotlib.pyplot as plt

#======================================================================
# From Prof. Yamaguchi
#======================================================================
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

# model = solver.prototxt
# base_weights = somthing.caffemodel
# gpu = 0 or int(os.getenv('SGE_GPU', 0))
# max_iter = argument to solver.step() such as solver.step(max_iter)
def solve(model, base_weights,out_model, gpu, max_iter):
    # init
    if gpu is not None:
        caffe.set_mode_gpu()
        caffe.set_device(int(gpu))
        print("Using GPU={0}".format(int(gpu)))
    else:
        caffe.set_mode_cpu()
    #solver = caffe.SGDSolver(os.path.join(MODEL_DIR, model, 'solver.prototxt'))
    solver = caffe.SGDSolver(model)
    # do net surgery to set the deconvolution weights for bilinear interpolation
    interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
    interp_surgery(solver.net, interp_layers)
    # copy base weights for fine-tuning
    if (base_weights!='None'):
        solver.net.copy_from(base_weights)
    # Hack to copy pre-trained weights between forks
    # solve straight through -- a better approach is to define a solving loop to
    # 1. take SGD steps
    # 2. score the model by the test net `solver.test_nets[0]`
    # 3. repeat until satisfied
    solver.step(max_iter)
    solver.net.save(out_model)
    del solver
#======================================================================
# END: From Prof. Yamaguchi
#======================================================================

def setup(options):
    # prototxt = 'TVG_CRFRNN_COCO_VOC.prototxt'
    # prototxt = 'train_val.prototxt'

    # solver_prototxt = 'solver.prototxt'
    solver_prototxt = options.solver
    # pretrained_model = '../fcn-pascalcontext/fcn-32s-pascalcontext.caffemodel'
    pretrained_model = options.base_weights
    out_model = options.out_model
    num_iter = options.iter

    SGE_GPU = int(os.getenv('SGE_GPU', 0))
    
    print 'Start training'

    if (os.path.exists(pretrained_model)):
        solve(solver_prototxt,pretrained_model,out_model,SGE_GPU,num_iter)
    else:
        print pretrained_model+' does not exists.'

if __name__ == "__main__":
    logging.basicConfig(format='[%(asctime)s] %(message)s',
                        level=logging.DEBUG)
    parser = argparse.ArgumentParser(
        description='Train a convolutional neural network for fashion-parsing.')
    parser.add_argument('--solver', type=str, required=True,
                        help='solver.prototxt file of the network')
    parser.add_argument('--base_weights', type=str, default=None,
                        help='caffemodel file to be used as based model')
    parser.add_argument('--out_model', type=str, required=True,
                        help='caffemodel file to be saved to')
    parser.add_argument('--iter', type=int, default=80000,
			help='number of training iteration, default 80,000')
    setup(parser.parse_args())
