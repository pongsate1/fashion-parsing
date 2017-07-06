Fashion FCN data v20151107
==========================

Download link

http://vision.is.tohoku.ac.jp/~kyamagu/private/longjon-caffe/fashionfcn-20151107.tgz


Contents
--------

The archive contains the following data.

    fcn-16s-fashionista-v0.2/
    fcn-16s-fashionista-v1.0/
    fcn-32s-fashionista-v0.2/
    fcn-32s-fashionista-v1.0/
    fcn-8s-fashionista-v0.2/
    fcn-8s-fashionista-v1.0/
    images/
    metadata.json
    result-fashionista-v0.2.json
    result-fashionista-v1.0.json
    truth-fashionista-v0.2/
    truth-fashionista-v1.0/

Various metadata are stored in json format.

### Dataset metadata

Dataset information is saved in `metadata.json`.

    {
      "fashioninsta-v0.2": {
        "labels": ["null", ...],
        "size": 685,
        "train": [...],
        "val": [...],
        "test": [...]
      },
      "fashioninsta-v1.0": {
        "labels": ["background", ...],
        "size": 685,
        "train": [...],
        "val": [...],
        "test": [...]
      },
    }

### Images and annotations

The images are located under `images/` with the image id as a filename, e.g.,

    images/1.jpg
    images/2.jpg

All images are in JPEG format.

Annotation data are saved in PNG format under `truth-$DATASET/` dir, e.g.,

    truth-fashionista-v0.2/1.png
    truth-fashionista-v0.2/2.png

The labels are encoded in each pixel as integer values. The corresponding label names should be found in the metadata.

    import json
    import numpy
    import PIL.Image

    # Load metadata.
    filename = 'fashionfcn/metadata.json'
    metadata = json.load(open(filename, 'r'))
    labels = metadata['fashionista-v0.2']['labels']

    # Load annotation.
    filename = 'fashionfcn/truth-fashionista-v0.2/1.png'
    image = numpy.asarray(PIL.Image.open(filename, 'r'))

    # Get labels.
    print labels[image[0,0]]  # label at pixel (0, 0) is 'null'


### FCN parsing results

The files `result-fashioniosta-v0.2.json` and `result-fashionista-v1.0.json`
are the metadata on evaluation result using fcn models.

    {
      "labels": ["null", ...],
      "results": [
        [
          {
            "model": "fcn-32s-fashionista-v0.2"
            "evaluation": {
              "accuracy": ...
            }
            "dataset": "train-gt.lmdb"
          },
          {
            "model": "fcn-32s-fashionista-v0.2"
            "evaluation": ...
            "dataset": "val-gt.lmdb"
          },
          {
            "model": "fcn-32s-fashionista-v0.2"
            "evaluation": ...
            "dataset": "test-gt.lmdb"
          }
        ],
        [
          {
            "model": "fcn-16s-fashionista-v0.2"
            "evaluation": ...
            "dataset": "train-gt.lmdb"
          },
          {
            "model": "fcn-16s-fashionista-v0.2"
            "evaluation": ...
            "dataset": "val-gt.lmdb"
          },
          {
            "model": "fcn-16s-fashionista-v0.2"
            "evaluation": ...
            "dataset": "test-gt.lmdb"
          }
        ],
        ...
      ]
    }

Using Python, the json files can be parsed by the following approach.

    import json
    data = json.load(open('fashionfcn/result-fashionista-v0.2.json','r'))
    data["labels"]         # text labels
    data["results"][0][2]  # testing evaluation (2) for model fcn-32s (0)

Individual results are available at the `fashionfcn/$MODEL-$DATASET/result.json`.

    import json
    data = json.load(open('fashionfcn/fcn-32s-fashionista-v0.2/result.json','r'))
    data[0]['examples'][0] # result for the first instance using model 0
