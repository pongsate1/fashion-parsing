'''
Export the given datasets.

python examples/fashionpose/export.py \
  --inputs data/fashionista-v1.0/test-1.h5 \
  --output public/fashionpose/fashionista-v1.0

python examples/fashionpose/export.py \
  --inputs public/fashionpose/poseseg-32s-fashionista-v1.0-test.h5 \
  --output public/fashionpose/poseseg-32s-fashionista-v1.0-test
'''

import argparse
import h5py
import logging
import numpy as np
import os
import PIL.Image


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


def createListTxt(options):
    '''
    Create text file containing filenames.
    '''
    logging.info("createListTxt")
    fimg_list = open(options.output+'/'+'img_list.txt','w')
    fgt_list = open(options.output+'/'+'gt_list.txt','w')
    fmask_list = open(options.output+'/'+'mask_list.txt','w')
    for input_file in options.inputs:
        assert os.path.exists(input_file)
        logging.info("Input: {0}".format(input_file))
        for output in progress(hdf5_reader(input_file), 100):
            if 'image' in output:
                fimg_list.write("{0}.{1}".format(output['id'][0], "jpg\n"))
            if 'segmentation' in output:
                fgt_list.write("{0}.{1}".format(output['id'][0], "png\n"))
            if 'seg_prob' in output:
                fmask_list.write("{0}.{1}".format(output['id'][0], "png\n"))
            if 'score' in output:
                fmask_list.write("{0}.{1}".format(output['id'][0], "png\n"))

def export(options):
    '''
    Export results to image files.
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
        for output in progress(hdf5_reader(input_file), 100):
            if 'image' in output:
                filename = os.path.join(options.output,"{0}.{1}".format(
                    output['id'][0], "jpg"))
                image_mean = np.array(options.image_mean).reshape(3,1,1)
                image = output['image'] + image_mean
                image = image[::-1,:,:].transpose((1,2,0)).astype('uint8')
                PIL.Image.fromarray(image).resize(
                    options.output_size, resample=PIL.Image.NEAREST
                    ).save(filename)
            if 'segmentation' in output:
                filename = os.path.join(options.output,"{0}.{1}".format(
                    output['id'][0], "png"))
                annotation = output['segmentation'][0,:].astype('uint8')
                PIL.Image.fromarray(annotation).resize(
                    options.output_size, resample=PIL.Image.NEAREST
                    ).save(filename)
                
            if 'seg_prob' in output:
                filename = os.path.join(options.output,"{0}.{1}".format(
                    output['id'][0], "png"))
                prediction = output['seg_prob'].argmax(axis=0).astype('uint8')
                PIL.Image.fromarray(prediction).resize(
                    options.output_size, resample=PIL.Image.NEAREST
                    ).save(filename)

            if 'score' in output:
                filename = os.path.join(options.output,"{0}.{1}".format(
                    output['id'][0], "png"))
                prediction = output['score'].argmax(axis=0).astype('uint8')
                PIL.Image.fromarray(prediction).resize(
                    options.output_size, resample=PIL.Image.NEAREST
                    ).save(filename)

    
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
    parser.add_argument('--output_size', default=(400, 600),
                        help='Output image size.')
    parser.add_argument('--image_mean', default=[
        104.00699, 116.66877, 122.67892], help='Output image size.')
    export(parser.parse_args())
    createListTxt(parser.parse_args())