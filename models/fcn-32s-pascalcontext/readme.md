---
name: FCN-32s Fully Convolutional Semantic Segmentation on PASCAL-Context
caffemodel: fcn-32s-pascalcontext.caffemodel
caffemodel_url: http://dl.caffe.berkeleyvision.org/fcn-32s-pascalcontext.caffemodel
sha1: adbbd504c280e2b8966fc32e32ada2a2ecf13603
gist_id: 80667189b218ad570e82
---

This is a model from the [paper](http://cs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf):

    Fully Convolutional Networks for Semantic Segmentation
    Jonathan Long, Evan Shelhamer, Trevor Darrell
    arXiv:1411.4038

This is the single stream, 32 pixel prediction stride version.

This model was trained for the PASCAL-context 59-class (60 including background) task. The final layer outputs scores for each class, which may be normalized via softmax or argmaxed to obtain per-pixel labels. The first label (index zero) is background, with the rest following the order given by the dataset authors.

The input is expected in BGR channel order, with the following per-channel mean subtracted:

    B 104.00698793 G 116.66876762 R 122.67891434

This is a pre-release: it requires unmerged PRs to run. It should be usable with the branch available at https://github.com/longjon/caffe/tree/future. Training ought to be possible with that code, but the original training scripts have not yet been ported.

This model obtains 35.1 mean I/U on PASCAL-Context val.
