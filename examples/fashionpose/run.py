'''
Run a trained FCN model to the given dataset.

python examples/fashionpose/run.py \
  --model_def models/poseseg-32s-fashionista-v1.0/deploy.prototxt \
  --model_weights \
    models/poseseg-32s-fashionista-v1.0/train-0.5_iter_0.caffemodel \
  --input data/fashionista-v1.0/test-0.5.h5 \
  --output public/fashionpose/poseseg-32s-fashionista-v1.0.h5 \
  --labels_clothing data/fashionista-v1.0/labels.txt \
  --labels_joint data/fashionista-v1.0/joints.txt \
  --gpu 0

'''

import argparse
import h5py
import json
import logging
import numpy as np
import os
import sys
caffe_root = '/ceph/tangseng/fashion-parsing/'
sys.path.insert(0,caffe_root+'python')
import caffe
sys.path.append('examples/fashionpose')
from fashion_evaluation import *


class NumpyAwareJSONEncoder(json.JSONEncoder):
    '''JSON Encoder for numpy data.
    '''
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            if obj.ndim == 1:
                return obj.tolist()
            else:
                return [self.default(obj[i]) for i in range(obj.shape[0])]
        return json.JSONEncoder.default(self, obj)


def hdf5_info(filename, fieldnames=None):
    '''
    Inspect the HDF5 file contents.
    '''
    with h5py.File(filename, 'r') as f:
        if not fieldnames:
            fieldnames = f.keys()
        return {name: f[name].shape for name in fieldnames}


def hdf5_reader(filename, fieldnames=None):
    '''
    Read HDF5 records.
    '''
    with h5py.File(filename, 'r') as f:
        if not fieldnames:
            fieldnames = f.keys()
        logging.info("Opening {0} for {1}".format(
            filename, ", ".join(fieldnames)))
        fields = {name: f[name] for name in fieldnames}
        for i in xrange(fields.values()[0].shape[0]):
            yield {name: field[i, :] for name, field in fields.items()}


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


def process_batch(net, batch):
    '''
    Calculate the posemap.
    '''
    for i in xrange(len(batch)):
        in_ = np.array(batch[i]['image'], dtype=np.float32)
        if i == 0:
            # shape for input (data blob is N x C x H x W), set data
            net.blobs['image'].reshape(len(batch), *in_.shape)
        net.blobs['image'].data[i, ...] = in_
    net.forward()
    outputs = []
    for i in xrange(len(batch)):
        outputs.append({name: net.blobs[name].data[i, :]
                        for name in net.outputs})
        if "id" in batch[i]:
            outputs[i]["id"] = batch[i]["id"]
    return outputs


def evaluate_batch(outputs, batch, results):
    '''
    Evaluate the performance.
    '''
    for i in xrange(len(outputs)):
        # Keep id.
        if 'id' in outputs[i]:
            if 'id' not in results:
                results['id'] = []
            results['id'].append(outputs[i]['id'])
        # Semantic segmentation.
        if 'seg_prob' in outputs[i] and 'segmentation' in batch[i]:
            truth = batch[i]['segmentation']
            prediction = outputs[i]['seg_prob'].argmax(axis=0)
            confusion = compute_confusion(
                truth.flatten(), prediction.flatten(),
                xrange(outputs[i]['seg_prob'].shape[0]))
            result = evaluate_fashion_confusion(confusion)
            if not 'segmentation' in results:
                results['confusion'] = np.zeros(confusion.shape)
                results['segmentation'] = []
            results['confusion'] += confusion
            results['segmentation'].append(result)
            # add IOU
            if not 'IOU' in results:
                results['IOU']=[]
            IOU = compute_IOU(
                truth.flatten(), prediction.flatten(),
                xrange(outputs[i]['seg_prob'].shape[0]))
            results['IOU'].append(IOU)
        # Pose estimation.
        if 'pose_prob' in outputs[i] and 'posemap' in batch[i]:
            truth = find_keypoints_in_posemap(batch[i]['posemap'])
            prediction = find_keypoints_in_posemap(outputs[i]['pose_prob'])
            if not 'pose' in results:
                results['pose'] = []
            results['pose'].append({'truth': truth, 'prediction': prediction})


def evaluate_summary(results):
    '''Evaluate the summary result.
    '''
    if 'confusion' in results:
        results['segmentation_total'] = evaluate_fashion_confusion(
            results['confusion'])
    if 'pose' in results:
        truth = np.array([result['truth'] for result
            in results['pose']]).transpose((1,2,0))
        prediction = np.array([result['prediction'] for result
            in results['pose']]).transpose((1,2,0))
        results['pose_total'] = {'pck': evaluate_pck(truth, prediction)}


def initialize_datasets(f, input_shapes, output_shapes):
    '''
    Initialize the output shape.
    '''
    data_size = input_shapes.values()[0][0]
    datasets = {}
    datasets["id"] = f.create_dataset("id", (data_size, 1), dtype="i4")
    for key, blob_shape in output_shapes.items():
        datasets[key] = f.create_dataset(
            key,
            [data_size] + list(blob_shape[1:]),
            dtype="f",
            chunks=blob_shape,
            compression="gzip",
            compression_opts=9)
    return datasets


def save_outputs(datasets, offset, outputs):
    '''
    Save the output blob to datasets.
    '''
    for i in xrange(len(outputs)):
        for key in outputs[i]:
            datasets[key][offset + i, ...] = outputs[i][key]


def run(options):
    '''
    Run a trained model to the given input.
    '''
    if not os.path.exists(os.path.dirname(options.output)):
        logging.info("Creating {0}".format(os.path.dirname(options.output)))
        os.makedirs(os.path.dirname(options.output))
    if os.path.exists(options.output):
        logging.info("Output exists: {0}".format(options.output))

    # Set up caffe.
    if options.gpu is not None:
        caffe.set_mode_gpu()
        caffe.set_device(int(options.gpu))
    else:
        caffe.set_mode_cpu()
    net = caffe.Net(options.model_def, options.model_weights, caffe.TEST)
    output_shapes = {name: net.blobs[name].data.shape for name in net.outputs}

    results = {}
    # Keep label data together.
    if options.labels_clothing:
        results["labels_clothing"] = [label.strip() for label in
                                      open(options.labels_clothing, "r")]
    if options.labels_joint:
        results["labels_joint"] = [label.strip() for label in
                                   open(options.labels_joint, "r")]
    imgMeanIOUs=[]
    # Process input.
    logging.info("Creating output: {0}".format(options.output))
    with h5py.File(options.output, "w") as f:
        logging.info("Input: {0}".format(options.input))
        input_shapes = hdf5_info(options.input)
        datasets = initialize_datasets(f, input_shapes, output_shapes)
        offset = 0
        for batch in each_batch(hdf5_reader(options.input),
                                size=options.batch_size):
            outputs = process_batch(net, batch)
            save_outputs(datasets, offset, outputs)
            evaluate_batch(outputs, batch, results)
            imgMeanIOU = results['IOU'][-1][-1]
            imgMeanIOUs = np.append(imgMeanIOUs, imgMeanIOU)
            offset += len(batch)
            if offset % 100 == 0:
                logging.info("Processed {0} of {1}".format(
                    offset, input_shapes.values()[0][0]))

    evaluate_summary(results)
    evaluate_summary(results)
    results['AllMeanIOU'] = np.mean(imgMeanIOUs)

    # Save the evaluation results.
    json_file = format(options.output).replace(".h5", ".json")
    logging.info("Saving {0}".format(json_file))
    with open(json_file, "w") as file_ptr:
        json.dump(results, file_ptr, cls=NumpyAwareJSONEncoder)


if __name__ == "__main__":
    logging.basicConfig(format='[%(asctime)s] %(message)s',
                        level=logging.DEBUG)
    parser = argparse.ArgumentParser(
        description='Run a trained Fashion-FCN model.')
    parser.add_argument('--model_def', type=str, required=True,
                        help='Model definition to test.')
    parser.add_argument('--model_weights', type=str, required=True,
                        help='Model weights to test.')
    parser.add_argument('--input', type=str, required=True,
                        help='Input data.')
    parser.add_argument('--output', type=str, required=True,
                        help='Output data.')
    parser.add_argument('--gpu', type=str, default=None,
                        help='GPU device id.')
    parser.add_argument('--labels_clothing', type=str, default=None,
                        help='Text file of clothing lables.')
    parser.add_argument('--labels_joint', type=str, default=None,
                        help='Text file of joint lables.')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size.')
    run(parser.parse_args())
