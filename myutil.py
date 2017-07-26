import os
import numpy as np
from PIL import Image as PILImage
import glob
from scipy import misc
import json
caffe_root = '/media/local_hdd/resnet/caffe/'
import sys
sys.path.append(caffe_root + 'python')
import caffe
import matplotlib.pyplot as plt
import h5py
import logging
from PIL import Image
logging.basicConfig(format='%(asctime)s %(message)s',level=logging.INFO)

DATA_DIR = 'data/'
MODEL_DIR = 'models/'

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

color_code = [[255,255,255],[226,196,196],[64,32,32],[255,0,0],[255,70,0],[255,139,0],[255,209,0],[232,255,0],[162,255,0],[93,255,0],[23,255,0],[0,255,46],[0,255,116],[0,255,185],[0,255,255],[0,185,255],[0,116,255],[0,46,255],[23,0,255],[93,0,255],[162,0,255],[232,0,255],[255,0,209],[255,0,139],[255,0,70]]

pallete = sum(color_code,[])

color_code = [[255,255,255],[255,67,0],[255,133,0],[255,200,0],[244,255,0],[177,255,0],[111,255,0],[44,255,0],[226,196,196],[64,32,32],[0,255,155],[0,255,222],[0,222,255],[0,155,255],[0,89,255],[0,22,255],[44,0,255],[206,176,176],[177,0,255],[244,0,255],[255,0,200],[255,0,133],[255,0,67]]

pallete_cfpd = sum(color_code,[])

# pallete_cfpd = [255,255,255, # bk
#             155,195,170, # t-shirt
#             64,100,100, # bag
#             128,128,0, # belt
#             0,0,128, # blazer
#             128,0,128, # blouse
#             0,128,128, # coat
#             128,128,128, # dress
#             230,200,180, # face
#             64, 32, 32, # hair
#             64,128,0, # hat
#             192,128,0, # jeans
#             64,0,128, # legging
#             192,0,128, # pants
#             64,128,128, # scarf
#             192,128,128, # shoe
#             0,64,0, # shorts
#             255,195,170, # skin
#             0,192,0, # skirt
#             128,192,0, # socks
#             0,64,128, # stocking
#             128,64,128, # sunglasses
#             0,192,128, # sweater
#             128,192,128,
#             64,64,0,
#             192,64,0,
#             64,192,0,
#             192,192,0,
#             100,0,0,
#             0,100,0,
#             0,0,100,
#             100,100,100]

image_mean = [104.00699, 116.66877, 122.67892]

def getFullFilenames(DIR,FILETYPE):
    '''
    get absolute path of files in DIR with FILETYPE
    '''
    filenames = []
    path = glob.glob(DIR+"*."+FILETYPE)
    for filename in path:
        filenames.append(filename)
    return np.sort(filenames)

def showTruth(truth,DATASET=''):
    '''
    make a label image of ground truth
    '''
    truth = np.uint8(truth)
    output_im = PILImage.fromarray(truth)
    if DATASET == 'cfpd' or DATASET == 'tmm_dataset_sharing':
        output_im.putpalette(pallete_cfpd)
    else:
        output_im.putpalette(pallete)
    return output_im
    
def showSegMap(segBlob,DATASET=''):
    '''
    Return segmented image
    '''
    labelmap = segBlob#.argmax(axis=0)
    labelmap = np.uint8(labelmap)
    output_im = PILImage.fromarray(labelmap)
    if DATASET == 'cfpd' or DATASET == 'tmm_dataset_sharing':
        output_im.putpalette(pallete_cfpd)
    else:
        output_im.putpalette(pallete)
    return output_im

def calAccuracy(segBlob,truth):
    labelmap = segBlob#.argmax(axis=0)
    acc = np.mean(labelmap[:]==truth[:])*100
    return acc

def calIOU(segmap,truth,nlable):
    '''
    Calculate classIOU and meanIOU from segmap and truth
    '''
    labelmap = segmap#.argmax(axis=0)
    intersectCount = np.zeros(nlable)
    unionCount = np.zeros(nlable)
    classIOU = np.zeros(nlable)
    for i in xrange(1,nlable):
        segClass = labelmap==i        
        truthClass = truth==i
        intersect = np.logical_and(segClass,truthClass) 
        intersectCount[i] = int(sum(sum(intersect)))
        union = np.logical_or(segClass,truthClass)
        unionCount[i] = int(sum(sum(union)))
        if unionCount[i]!=0:
            classIOU[i] = intersectCount[i]/unionCount[i]
        else:
            classIOU[i] = np.nan
    meanIOU = sum(intersectCount)/sum(unionCount)
    #print meanIOU
    classIOU = classIOU*100
    meanIOU = meanIOU*100
    return [classIOU,meanIOU]

def getImgID(absPath):
    filename = basename(absPath)
    imgID = os.path.splitext(filename)[0]
    return imgID

def getLabels(DATASET):
    DATASET_DIR = 'data/'+DATASET
    LABEL_FILE = DATASET_DIR+'/labels.txt'
    labels = [label.strip() for label in open(LABEL_FILE, "r")]
    labels = np.array(labels)
    return labels

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

def initialize_datasets(f, input_shapes, output_shapes):
    '''
    Initialize the output shape.
    '''
    data_size = input_shapes.values()[0][0]
    datasets = {}
    # datasets["id"] = f.create_dataset("id", (data_size, 1), dtype="i4")
    for key, blob_shape in output_shapes.items():
        datasets[key] = f.create_dataset(
            key,
            [data_size] + list(blob_shape[1:]),
            dtype="f",
            chunks=blob_shape,
            compression="gzip",
            compression_opts=9)
    return datasets

def new_initialize_datasets(f, num_data, output_shapes):
    '''
    Initialize the output shape.
    '''
    data_size = num_data
    datasets = {}
    # datasets["id"] = f.create_dataset("id", (data_size, 1), dtype="i4")
    for key, blob_shape in output_shapes.items():
        datasets[key] = f.create_dataset(
            key,
            [data_size] + list(blob_shape[1:]),
            dtype="f",
            chunks=blob_shape,
            compression="gzip",
            compression_opts=9)
    return datasets

def imageFromH5(h5Image,image_mean=image_mean):
    image = h5Image
    image_mean = np.array(image_mean).reshape(3,1,1)
    image = image + image_mean
    image = image[::-1,:,:].transpose((1,2,0)).astype('uint8')

    return image

def saveDatasetToH5(dataset,outH5Path):
    fieldnames = dataset.keys()
    logging.info ("Saving to "+outH5Path)
    with h5py.File(outH5Path, 'w') as f:
        for key in fieldnames:
            f.create_dataset(key,data=dataset[key],compression="gzip",compression_opts=9)

def setupCNN(CAFFEMODEL,DEPLOY_PROTOTXT,SGE_GPU=0):
    logging.info("Setting up CNN network.")
    SGE_GPU = int(os.getenv('SGE_GPU', SGE_GPU))
    caffe.set_device(SGE_GPU)
    caffe.set_mode_gpu()
    net = caffe.Net(DEPLOY_PROTOTXT,CAFFEMODEL,caffe.TEST)
    return net

def setupCNN_cpu(CAFFEMODEL,DEPLOY_PROTOTXT,SGE_GPU=0):
    logging.info("Setting up CNN network.")
    SGE_GPU = int(os.getenv('SGE_GPU', SGE_GPU))
    caffe.set_mode_cpu()
    net = caffe.Net(DEPLOY_PROTOTXT,CAFFEMODEL,caffe.TEST)
    return net

def setupTransformer(mean_vec = np.array([104.00699, 116.66877, 122.67892],dtype=np.float32),dim=(1,3,224,224)):
    '''
    Input: CNN, RGB values
    Output: a transformer for processing raw image to image for CNN    
    '''
    transformer = caffe.io.Transformer({'data': dim})
    transformer.set_transpose('data', (2,0,1))
    transformer.set_mean('data', mean_vec) # mean pixel
    transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
    transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB
    return transformer

def readH5toVar(h5Path,nelement=-1):
    '''
    Read h5 data into a variable
    '''
    h5reader = hdf5_reader(h5Path)
    dataHolder = {}
    count = 0
    for elem in h5reader:
        for key in elem.keys():
            if not key in dataHolder:
                dataHolder[key] = []
            dataHolder[key].append(elem[key])
        
        if nelement > 0:
            count+=1
            if count > nelement:
                break
    return dataHolder