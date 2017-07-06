'''
Load FCNs output h5 and evaluate to ground truth.

python python/fromNetOutH5toJson.py \
  --input models/fcn-8s-fashionista-v1.0/test-iter80000.h5 \
  --output models/fcn-8s-fashionista-v1.0/test-iter80000.json \
  --gt data/fashionista-v1.0/test-1.h5/ \
  --labels_clothing data/fashionista-v1.0/labels.txt \
'''

# import argparse
# import h5py
# import json
# import logging
# import numpy as np
# import os
# import sys
# import glob
# caffe_root = '/ceph/tangseng/fashion-parsing/'
# sys.path.insert(0,caffe_root+'python')
# import caffe
# sys.path.append('examples/fashionpose')
# from fashion_evaluation import *
# from scipy import misc
# from os.path import basename
from crf_email_eval import *

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


def run(options):
    
    logging.info("Exporting {0}".format(options.output))
    logging.info("Ground truth dir {0}".format(options.gt))
    GT_DIR = options.gt
    if GT_DIR[-1]!='/':
        GT_DIR.append('/')

    input_file = options.input

    results = {} # used to create json file.
    labels = []
    # Keep label data together.
    if options.labels_clothing:
        results["labels_clothing"] = [label.strip() for label in
                                      open(options.labels_clothing, "r")]
        label_file =  open(options.labels_clothing, "r")
        for line in label_file:
            labels.append(line)
    
    imgMeanIOUs=[]
    
    for h5data in progress(hdf5_reader(input_file), 100):
        imgID = str(h5data['id'][0])
        # print imgID
        if 'seg_prob' in h5data:
            prediction = h5data['seg_prob'].argmax(axis=0).astype('uint8')
        if 'score' in h5data:
            prediction = h5data['score'].argmax(axis=0).astype('uint8')


        gt_img_filename = GT_DIR+str(imgID)+'.png'
        if os.path.exists(gt_img_filename) == False:
            print "ERROR: Ground truth image "+gt_img_filename+" not found!!!"
            exit()

        gtImg = misc.imread(gt_img_filename)
        evaluate_img(imgID,gtImg,prediction,len(labels),results)
        imgMeanIOU = results['IOU'][-1][-1]
        imgMeanIOUs = np.append(imgMeanIOUs, imgMeanIOU)
        print "ImgID: "+str(imgID)+"\tImgMeanIOU = "+str(imgMeanIOU) #+ "\tEvaluation Complete: "+str(i+1)+"/"+str(numImg)+" images"

    evaluate_summary(results)
    results['AllMeanIOU'] = np.mean(imgMeanIOUs)
    print "Average IOU"+str(np.mean(imgMeanIOUs))

    # print len(crf_refine_img_filenames),len(gt_img_filenames)
    json_file = format(options.output)
    logging.info("Saving {0}".format(json_file))
    with open(json_file, "w") as file_ptr:
        json.dump(results, file_ptr, cls=NumpyAwareJSONEncoder)


if __name__ == "__main__":
    logging.basicConfig(format='[%(asctime)s] %(message)s',
                        level=logging.DEBUG)
    parser = argparse.ArgumentParser(
        description='Load output of FCNs (.h5) and create appropiate JSON.')
    parser.add_argument('--input', type=str, required=True,
                        help='Input h5 fle.')
    parser.add_argument('--output', type=str, required=True,
                        help='Output JSON file.')
    parser.add_argument('--gt', type=str, required=True,
                        help='ground truth png dir')
    parser.add_argument('--labels_clothing', type=str, default=None,
                        help='Text file of clothing labels.')
    
    run(parser.parse_args())
