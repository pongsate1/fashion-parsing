'''
Export the given datasets.

python examples/fashionfcn/export.py \
  --inputs data/fashionista-v1.0/train.lmdb \
           data/fashionista-v1.0/val.lmdb \
           data/fashionista-v1.0/test.lmdb \
  --output public/fashionfcn/images \
  --format jpg

python examples/fashionfcn/export.py \
  --inputs data/fashionista-v1.0/train-gt.lmdb \
           data/fashionista-v1.0/val-gt.lmdb \
           data/fashionista-v1.0/test-gt.lmdb \
  --output public/fashionfcn/truth-fashionista-v1.0

'''

import argparse
import lmdb
import logging
import os
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


def progress(iterable, num):
    '''
    Print the current progress.
    '''
    i = 0
    for iterator in iterable:
        yield iterator
        i = i + 1
        if (i % num) == 0:
            logging.info("{0}".format(i))
    logging.info("{0}".format(i))


def export(options):
    '''
    Run a trained model to the given input.
    '''
    # Make the directory.
    logging.info("Exporting {0}".format(options.output))
    if os.path.exists(options.output):
        logging.info("Exists: {0}".format(options.output))
    else:
        logging.info("Creating {0}".format(options.output))
        os.makedirs(options.output)

    # Process each input.
    for input_file in options.inputs:
        assert os.path.exists(input_file)
        logging.info("Input: {0}".format(input_file))
        ids = []
        for key, value in progress(lmdb_reader(input_file), 100):
            filename = os.path.join(options.output,
                                    "{0}.{1}".format(key, options.format))
            image = caffe.proto.caffe_pb2.Datum.FromString(value).data
            with open(filename, 'wb') as file:
                file.write(image)
            ids.append(int(key))
    logging.info("Done.")


if __name__ == "__main__":
    logging.basicConfig(format='[%(asctime)s] %(message)s',
                        level=logging.DEBUG)
    parser = argparse.ArgumentParser(description='Export the images.')
    parser.add_argument('--inputs', nargs='+',
                        help='Input data.')
    parser.add_argument('--output', type=str, required=True,
                        help='Output data.')
    parser.add_argument('--output_metadata', type=str, default=None,
                        help='Output metadata.')
    parser.add_argument('--format', type=str, default="png",
                        help='Image format, default png.')
    export(parser.parse_args())
