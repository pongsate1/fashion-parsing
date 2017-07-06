import glob
import json
import lmdb
import logging
import numpy as np
from PIL import Image
import os
import re
import sklearn.metrics
import StringIO
import sys
sys.path.append('python')
import caffe

logging.basicConfig(format='[%(asctime)s] %(message)s', level=logging.DEBUG)

image_db_file = 'data/fashionista-v1.0/fashionista-v1.0-test.lmdb'
label_db_file = 'data/fashionista-v1.0/fashionista-v1.0-test-gt.lmdb'
labels = [line.strip() for line in open('data/fashionista-v1.0/labels.txt')]
arch = 'fcn-32s-fashionista-v1.0'
model_files = glob.glob("models/{0}/train_iter_*.caffemodel".format(arch))
models = [re.findall(r'train_(iter_\d+).caffemodel', model)[0]
          for model in model_files]
output_dir = os.path.join('public', arch)

'''
def render_index():
    keys = [key for key, value in lmdb_reader(image_db_file)]
    model_names = [re.findall(r'train_(iter_\d+).caffemodel', model)[0]
                   for model in models]
    filename = os.path.join('tmp', 'fcn-32s-fashionista', 'index.html')
    logging.info("Rendering {0}".format(filename))
    with open(os.path.join('tmp', 'fcn-32s-fashionista', 'legend.json'), 'r') as f:
        legend = json.load(f)
    template = jinja2.Template("")
    template.stream(arch="fcn-32s-fashionista",
                    legend=legend,
                    keys=keys[0:30],
                    model_names=model_names).dump(filename)
'''


class NumpyAwareJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            if obj.ndim == 1:
                return obj.tolist()
            else:
                return [self.default(obj[i]) for i in range(obj.shape[0])]
        return json.JSONEncoder.default(self, obj)


def lmdb_reader(filename, **kwargs):
    with lmdb.open(filename, readonly=True, create=False, lock=False, **kwargs) as env:
        with env.begin() as txn:
            with txn.cursor() as cursor:
                for key, value in cursor:
                    yield key, value


def compute_segmentation(net, images):
    '''
    Calculate the segmentation.
    '''
    for i in xrange(len(images)):
        # Switch to BGR, subtract mean, and make dims C x H x W for Caffe
        in_ = np.array(images[i], dtype=np.float32)
        in_ = in_[:, :, ::-1]
        in_ -= np.array((104.00698793, 116.66876762, 122.67891434))
        in_ = in_.transpose((2, 0, 1))
        if i == 0:
            # shape for input (data blob is N x C x H x W), set data
            net.blobs['data'].reshape(len(images), *in_.shape)
        net.blobs['data'].data[i, ...] = in_
    # run net and take argmax for prediction
    net.forward()
    segmentations = []
    for i in xrange(len(images)):
        prediction = net.blobs['score'].data[i].argmax(axis=0).astype(np.uint8)
        segmentations.append(prediction)
    return segmentations


def decode_images(chunk):
    '''
    Decode chunks of encoded data.
    '''
    return [Image.open(StringIO.StringIO(
            caffe.proto.caffe_pb2.Datum.FromString(value).data))
            for key, value in chunk]


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
    accuracy = np.sum(np.diag(confusion)).astype(np.float64) / np.sum(confusion.flatten())
    precision = np.divide(np.diag(confusion).astype(np.float64), np.sum(confusion, axis=0))
    recall = np.divide(np.diag(confusion).astype(np.float64), np.sum(confusion, axis=1))
    f1 = np.divide(2 * np.diag(confusion).astype(np.float64),
                   np.sum(confusion, axis=0) + np.sum(confusion, axis=1))
    result = {
        'accuracy': float(accuracy),
        'average_precision': float(np.nanmean(precision)),
        'average_recall': float(np.nanmean(recall)),
        'average_f1': float(np.nanmean(f1)),
        'precision': precision.tolist(),
        'recall': recall.tolist(),
        'f1': f1.tolist()
        }
    return result


def each_chunk(iterable, size=16):
    '''
    Make a chunk from iterable.
    '''
    chunk = []
    for value in iterable:
        chunk.append(value)
        if len(chunk) % size == 0:
            yield chunk
            chunk = []
    if len(chunk) > 0:
        yield chunk


def progress(iterable, chunk=1000, message=None):
    '''
    Show progress for every chunk processed.
    '''
    if message is None:
        message = lambda i: "{0} processed.".format(i)
    i = 0
    for value in iterable:
        yield value
        i += 1
        if i % chunk == 0:
            logging.info(message(i))


def export_images(image_db_file, image_output_dir, format="jpg"):
    '''
    Export images in the LMDB to a directory.
    '''
    if not os.path.exists(image_output_dir):
        logging.info("Creating {0}".format(image_output_dir))
        os.makedirs(image_output_dir)
        for key, value in lmdb_reader(image_db_file):
            image_file = os.path.join(image_output_dir, "{0}.{1}".format(key, format))
            if os.path.exists(image_file):
                continue
            datum = caffe.proto.caffe_pb2.Datum.FromString(value)
            logging.info("Exporting {0}".format(image_file))
            with open(image_file, 'wb') as file:
                file.write(datum.data)


def run_testing():
    '''
    Run testing.
    '''
    caffe.set_device(int(os.getenv('SGE_GPU', 0)))
    caffe.set_mode_gpu()
    model_definition = "models/{0}/deploy.prototxt".format(arch)
    for k in xrange(len(models)):
        logging.info("Model: {0}".format(models[k]))
        net = caffe.Net(model_definition, model_files[k], caffe.TEST)
        model_output_dir = os.path.join(output_dir, models[k])
        if not os.path.exists(model_output_dir):
            os.makedirs(model_output_dir)
        for chunk in each_chunk(lmdb_reader(image_db_file), size=4):
            keys = [key for key, value in chunk]
            parse_files = [os.path.join(model_output_dir, "{0}.png".format(key))
                           for key in keys]
            logging.info("Computing segmentation for [{0}]".format(
                ", ".join(keys)))
            images = decode_images(chunk)
            segmentations = compute_segmentation(net, images)
            for i in xrange(len(segmentations)):
                logging.info("Saving {0}".format(parse_files[i]))
                Image.fromarray(segmentations[i]).save(parse_files[i])
        del net
    logging.info("Done.")


def evaluate_results():
    '''
    Evaluate the parsed images.
    '''
    result = {}
    total_confusions = np.zeros((len(labels), len(labels), len(models)))
    for key, datum in lmdb_reader(label_db_file):
        annotation = Image.open(StringIO.StringIO(
            caffe.proto.caffe_pb2.Datum.FromString(datum).data))
        result[key] = []
        accuracies = []
        for k in xrange(len(models)):
            parse_file = os.path.join(output_dir,
                                      "{0}".format(models[k]),
                                      "{0}.png".format(key))
            prediction = Image.open(parse_file)
            confusion = compute_confusion(annotation, prediction, labels)
            evaluation = evalute_performance(confusion)
            result[key].append(evaluation)
            total_confusions[:, :, k] = total_confusions[:, :, k] + confusion
            accuracies.append(evaluation['accuracy'])
        logging.info("{0:>3}: ".format(key) + ", ".join(map(lambda x: "{0:.2f}".format(x), accuracies)))
    result_file = os.path.join(output_dir, 'result.json')
    total_result = [evalute_performance(total_confusions[:, :, k])
                    for k in xrange(len(models))]
    logging.info("Exporting {0}".format(result_file))
    with open(result_file, 'w') as json_file:
        json.dump({'labels': labels,
                   'models': models,
                   'total_evaluation': total_result,
                   'evaluation': result},
                  json_file,
                  cls=NumpyAwareJSONEncoder)
    logging.info("Done.")


def create_output_directory(output_dir):
    '''
    Create output file structure if not existing.
    '''
    if not os.path.exists(output_dir):
        logging.info("Creating {0}".format(output_dir))
        os.makedirs(output_dir)
        for html_file in glob.glob("models/{0}/*.html".format(arch)):
            os.symlink(os.path.join("..", "..", html_file),
                       os.path.join(output_dir, os.path.basename(html_file)))
        # os.symlink(os.path.join("..", "fcn-32s-fashionista-v1.0", "images"),
        #            os.path.join(output_dir, "images"))
        # os.symlink(os.path.join("..", "fcn-32s-fashionista-v1.0", "annotations"),
        #            os.path.join(output_dir, "annotations"))
    export_images(image_db_file, os.path.join(output_dir, 'images'), 'jpg')
    export_images(label_db_file, os.path.join(output_dir, 'annotations'), 'png')


if __name__ == "__main__":
    create_output_directory(output_dir)
    run_testing()
    evaluate_results()
