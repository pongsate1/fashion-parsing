import sys
sys.path.append("python")
import caffe.io
from caffe.proto.caffe_pb2 import Datum
from StringIO import StringIO
from PIL import Image
import h5py
import leveldb
import lmdb
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid

INPUT_PATH = "data/fashionista-v1.0/val.lmdb"
RESULT_PATH = "public/fashionpose/pose-32s-fashionista-v1.0.h5"
ANNOTATION_PATH = "data/fashionista-v1.0/val-pose.leveldb"
ID = "135"
JOINT_NAMES = [line.strip() for line in open('data/fashionista-v1.0/joints.txt', 'r')]


# Fetch image.
def get_image(path, id):
    with lmdb.open(path, readonly=True, lock=False) as env:
        with env.begin() as txn:
            datum_string = txn.get(id)
    if not datum_string:
        print "Not found {0}".format(id)
        return None
    return np.array(Image.open(StringIO(Datum.FromString(datum_string).data)),
                    dtype="uint8")


# Fetch result.
def get_posemap(path, id):
    file = h5py.File(path)
    return file[ID].value


# Fetch image.
def get_annotation(path, id):
    db = leveldb.LevelDB(path)
    datum_string = db.Get(id)
    if not datum_string:
        print "Not found {0}".format(id)
        return None
    return caffe.io.datum_to_array(Datum.FromString(datum_string))


# Display the result.
def main():
    image = get_image(INPUT_PATH, ID)
    posemap = get_posemap(RESULT_PATH, ID)
    fig = plt.figure(1)
    grid = ImageGrid(fig, 111, nrows_ncols=(3, 5), axes_pad=0.1)
    grid[0].imshow(image)
    grid[0].axis("off")
    for i in xrange(posemap.shape[0]):
        grid[i+1].imshow(posemap[i, :, :], cmap="gray", interpolation="nearest")
        grid[i+1].axis("off")
        grid[i+1].text(5, 50, JOINT_NAMES[i], fontsize=10, color="white")
    plt.savefig('tmp/show_posemap.png', bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()
