'''
Evaluate the FCN segmentation results.

python examples/fashionfcn/eval.py \
  --inputs=data/fashionista-v1.0/train-gt.lmdb \
           data/fashionista-v1.0/val-gt.lmdb \
           data/fashionista-v1.0/test-gt.lmdb \
  --predictions=public/fashionfcn/fcn-32s-fashionista-v1.0 \
                public/fashionfcn/fcn-16s-fashionista-v1.0 \
                public/fashionfcn/fcn-8s-fashionista-v1.0 \
  --labels=data/fashionista-v1.0/labels.txt \
  --output=public/fashionfcn/evaluation.json

'''

import argparse
import json
import lmdb
import logging
import numpy as np
from PIL import Image
import os
import sklearn.metrics
import StringIO
import sys
sys.path.append('python')
import caffe


class NumpyAwareJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            if obj.ndim == 1:
                return obj.tolist()
            else:
                return [self.default(obj[i]) for i in range(obj.shape[0])]
        return json.JSONEncoder.default(self, obj)


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


def decode_images(batch):
    '''
    Decode chunks of encoded data.
    '''
    return [Image.open(StringIO.StringIO(
            caffe.proto.caffe_pb2.Datum.FromString(value).data))
            for key, value in batch]


def compute_confusion(annotation, prediction, labels):
    '''
    Compute a confusion matrix.
    '''
    truth = np.array(annotation, dtype=np.uint8).flatten()
    prediction = np.array(prediction, dtype=np.uint8).flatten()
    assert len(truth) == len(prediction)
    return sklearn.metrics.confusion_matrix(truth, prediction,
                                            [i for i in xrange(len(labels))])


def evalute_performance(confusion):
    '''
    Evaluate various performance measures from the confusion matrix.
    '''
    accuracy = (np.sum(np.diag(confusion)).astype(np.float64) /
                np.sum(confusion.flatten()))
    precision = np.divide(np.diag(confusion).astype(np.float64),
                          np.sum(confusion, axis=0))
    recall = np.divide(np.diag(confusion).astype(np.float64),
                       np.sum(confusion, axis=1))
    f1 = np.divide(2 * np.diag(confusion).astype(np.float64),
                   np.sum(confusion, axis=0) + np.sum(confusion, axis=1))
    # Remove the background from consideration.
    fg_accuracy = (np.sum(np.diag(confusion)[1:]).astype(np.float64) /
                   np.sum(confusion[1:,:].flatten()))
    result = {
        'accuracy': float(accuracy),
        'average_precision': float(np.nanmean(precision)),
        'average_recall': float(np.nanmean(recall)),
        'average_f1': float(np.nanmean(f1)),
        'fg_accuracy': float(fg_accuracy),
        'precision': precision.tolist(),
        'recall': recall.tolist(),
        'f1': f1.tolist()
        }
    return result


def evaluate_all(options):
    '''
    Evaluate the performance of predictions.
    '''
    # Load labels.
    assert os.path.exists(options.labels),(
           "Not found: {0}".format(options.labels))
    labels = [line.strip() for line in open(options.labels)]

    # Process each input.
    summary = {'results': [], 'labels': labels}
    for model in options.predictions:
        assert os.path.exists(model),(
               "Not found: {0}".format(model))
        logging.info("Model: {0}".format(model))
        results = []
        for input_file in options.inputs:
            logging.info("Data: {0}".format(input_file))
            evaluations = []
            total_confusion = np.zeros((len(labels), len(labels)))
            for key, value in lmdb_reader(input_file):
                parse_file = os.path.join(model,
                                          "{0}.png".format(key))
                assert os.path.exists(parse_file), (
                    "Not found: {0}".format(parse_file))
                prediction = np.asarray(Image.open(parse_file))
                annotation = Image.open(StringIO.StringIO(
                    caffe.proto.caffe_pb2.Datum.FromString(value).data))
                confusion = compute_confusion(annotation, prediction, labels)
                evaluation = evalute_performance(confusion)
                evaluation['id'] = key
                evaluations.append(evaluation)
                total_confusion = total_confusion + confusion
            total_evaluation = evalute_performance(total_confusion)
            results.append({
                'model': os.path.basename(model),
                'dataset': os.path.basename(input_file),
                'labels': labels,
                'evaluation': total_evaluation,
                'examples': evaluations
            })
        # Export results.
        json_file = os.path.join(model, 'result.json')
        logging.info("Writing {0}".format(json_file))
        json.dump(results, open(json_file, 'w'), cls=NumpyAwareJSONEncoder)
        # Keep summary.
        for i in xrange(len(results)):
            del results[i]['examples']
            del results[i]['labels']
        summary['results'].append(results)
    # Make summary across models.
    logging.info("Writing {0}".format(options.output))
    json.dump(summary, open(options.output, 'w'), cls=NumpyAwareJSONEncoder)


if __name__ == "__main__":
    logging.basicConfig(format='[%(asctime)s] %(message)s',
                        level=logging.DEBUG)
    parser = argparse.ArgumentParser(description='Run a trained FCN model.')
    parser.add_argument('--inputs', nargs='+',
                        help='Input data.')
    parser.add_argument('--predictions', nargs='+',
                        help='Prediction data.')
    parser.add_argument('--labels', type=str, required=True,
                        help='Text file that lists label names.')
    parser.add_argument('--output', type=str, required=True,
                        help='Output data.')
    evaluate_all(parser.parse_args())
