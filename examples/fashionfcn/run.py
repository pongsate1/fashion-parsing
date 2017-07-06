'''
Run a trained FCN model to the given dataset.

python examples/fashionfcn/run.py \
  --model_def=models/fcn-32s-fashionista-v1.0/deploy.prototxt \
  --model_weights=models/fcn-32s-fashionista-v1.0/train_iter_80000.caffemodel \
  --input=data/fashionista-v1.0/fashionista-v1.0-test.lmdb \
  --output=public/fashionfcn/fcn-32s-fashionista-v1.0/ \
  --gpu=0

'''

import argparse
import lmdb
import logging
import numpy as np
from PIL import Image
import os
import StringIO
import sys
sys.path.append('python')
import caffe


def lmdb_reader(filename, **kwargs):
    '''
    Read LMDB records.
    '''
    with lmdb.open(filename,
                   readonly=True,
                   create=False,
                   lock=False,
                   **kwargs) as env:
        with env.begin() as txn:
            with txn.cursor() as cursor:
                for key, value in cursor:
                    yield key, value


def each_batch(iterable, size=16):
    '''
    Make a batch from iterable.
    '''
    batch = []
    for value in iterable:
        batch.append(value)
        if len(batch) % size == 0:
            yield batch
            batch = []
    if len(batch) > 0:
        yield batch


def compute_segmentation(net, images, pixel_mean):
    '''
    Calculate the segmentation.
    '''
    for i in xrange(len(images)):
        # Switch to BGR, subtract mean, and make dims C x H x W for Caffe
        in_ = np.array(images[i], dtype=np.float32)
        in_ = in_[:, :, ::-1]
        in_ -= np.array(pixel_mean)
        in_ = in_.transpose((2, 0, 1))
        if i == 0:
            # shape for input (data blob is N x C x H x W), set data
            net.blobs['data'].reshape(len(images), *in_.shape)
        net.blobs['data'].data[i, ...] = in_
    # run net and take argmax for prediction
    net.forward()
    segmentations = []
    for i in xrange(len(images)):
        scores = net.blobs['score'].data[i]
        segmentations.append({
            'scores': scores,
            'prediction': scores.argmax(axis=0).astype(np.uint8)
            })
    return segmentations


def decode_images(batch):
    '''
    Decode chunks of encoded data.
    '''
    return [Image.open(StringIO.StringIO(
            caffe.proto.caffe_pb2.Datum.FromString(value).data))
            for key, value in batch]


def run(options):
    '''
    Run a trained model to the given input.
    '''
    if os.path.exists(options.output):
        logging.info("Output exists: {0}".format(options.output))
    else:
        logging.info("Creating output: {0}".format(options.output))
        os.makedirs(options.output)

    # Set up caffe.
    if options.gpu is not None:
        caffe.set_mode_gpu()
        caffe.set_device(int(options.gpu))
    else:
        caffe.set_mode_cpu()
    net = caffe.Net(options.model_def, options.model_weights, caffe.TEST)

    # Process each input.
    for input_file in options.input:
        logging.info("Input: {0}".format(input_file))
        for batch in each_batch(lmdb_reader(input_file), size=4):
            keys = [key for key, value in batch]
            logging.info("Computing segmentation for [{0}]".format(
                ", ".join(keys)))
            images = decode_images(batch)
            segmentations = compute_segmentation(net,
                                                 images,
                                                 options.model_mean)
            # Save results.
            for i in xrange(len(segmentations)):
                parse_file = os.path.join(options.output,
                                          "{0}.png".format(keys[i]))
                logging.info("Saving {0}".format(parse_file))
                image = Image.fromarray(segmentations[i]['prediction'])
                image.save(parse_file)
    del net


if __name__ == "__main__":
    logging.basicConfig(format='[%(asctime)s] %(message)s',
                        level=logging.DEBUG)
    parser = argparse.ArgumentParser(description='Run a trained FCN model.')
    parser.add_argument('--model_def', type=str, required=True,
                        help='Model definition to test.')
    parser.add_argument('--model_weights', type=str, required=True,
                        help='Model weights to test.')
    parser.add_argument('--model_mean', nargs=3, type=float,
                        default=[104.00698793, 116.66876762, 122.67891434],
                        help='Mean pixel value, default "104.00698793 ' +
                             '116.66876762 122.67891434".')
    parser.add_argument('--input', nargs='+',
                        help='Input data.')
    parser.add_argument('--output', type=str, required=True,
                        help='Output data.')
    parser.add_argument('--gpu', type=str, default=None,
                        help='GPU device id')
    run(parser.parse_args())
