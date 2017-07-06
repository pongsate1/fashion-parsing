'''
Export the given datasets.

python examples/fashionfcn/export_index.py \
  --datasets data/fashionista-v0.2/ \
             data/fashionista-v1.0/ \
  --splits train val test \
  --labels labels.txt \
  --output public/fashionfcn/metadata.json

'''

import argparse
import json
import lmdb
import logging
import os


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


def export(options):
    '''
    Export data.
    '''
    metadata = {}
    for dataset in options.datasets:
        label_file = os.path.join(dataset, options.labels)
        assert os.path.exists(label_file)
        dataname = os.path.basename(dataset)
        metadata[dataname] = {
            'labels': [line.strip() for line in open(label_file, 'r')]
        }
        size = 0
        for split in options.splits:
            database = os.path.join(dataset, "{0}.lmdb".format(split))
            logging.info("Reading {0}".format(database))
            ids = [int(key) for key, value in lmdb_reader(database)]
            split = os.path.basename(database).split('.')[0]
            metadata[dataname][split] = ids
            size = size + len(ids)
        metadata[dataname]['size'] = size
    logging.info("Writing {0}".format(options.output))
    json.dump(metadata, open(options.output, 'w'))


if __name__ == "__main__":
    logging.basicConfig(format='[%(asctime)s] %(message)s',
                        level=logging.DEBUG)
    parser = argparse.ArgumentParser(description='Export the metadata.')
    parser.add_argument('--datasets', nargs='+',
                        help='Input data.')
    parser.add_argument('--splits', nargs='+',
                        help='Split names.')
    parser.add_argument('--labels', type=str, default="labels.txt",
                        help='Label file name.')
    parser.add_argument('--output', type=str, required=True,
                        help='Output data.')
    export(parser.parse_args())
