Fashion-parsing 
===============

If you use this work, please cite https://arxiv.org/abs/1703.01386

This work extends fully-convolutional neural networks (FCN) for the clothing parsing problem.  

We extend FCN architecture with a side-branch network which we refer outfit encoder to predict a consistent set of clothing labels to encourage combinatorial preference, and with conditional random field (CRF) to explicitly consider coherent label assignment to the given image.

Live demo at http://vision.is.tohoku.ac.jp/clothing_parsing

Project page http://vision.is.tohoku.ac.jp/~tangseng/clothing_parsing_project

Contents
--------

1. Data

	Data is in data/. There are three fashion datasets: fashionista-v0.2, fashionista-v1.0, and tmm_dataset_sharing. See the instruction below for data preparation.

2. Models

	Models are in models/. There are 5 models used in fashion parsing: FCN-32s, FCN-16s, FCN-8s, Attribute Layers Training (codename: segc-8s-pre), Attribute Broadcast (codename: sege-8s), and Attribute filtering (codename: attrlog). The folder names are in <modelname>-<dataset> format. See the instruction below for training and running the model.

3. Parsing output and evaluation result

	Evaluation results and symbolic links to parsing output are in /public/fashionpose. This folder will be created automatically when run the model. Evaluation results are in json format. The actual output files of Attribute Broadcast (codename: sege-8s), and Attribute filtering (codename: attrlog) model are in the model's folder.

4. Script

	Python script and shell script are in examples/tangseng folder.

Instruction for fashion parsing
-------------------------------

1. Setup following environment: Python, Caffe, and MATLAB

2. Data preparation
	Download and convert data into appropiate format according to README and script in each dataset's directory under data/.
	
3. Download fcn-32s-pascalcontext.caffemodel according to th url in models/fcn-32s-pascalcontext/readme.md. This model is used as based model for training FCN-32s for fashion datasets.

4. Train FCN-32s, FCN-16s, FCN-8s, Attribute Layers Training (codename: segc-8s-pre), Attribute Broadcast (codename: sege-8s), and Attribute filtering (codename: attrlog) by execute:

	./examples/tangseng/train_all.sh
	
5. Run Attribute broadcast (sege) or Attribute filtering (attrlog) network by execute:

	./examples/tangseng/run_all.sh

	The output will be in models/<modelname>-<dataset>/. h5 segmentation output and json evaluation result are expected.

6. Prepare data for smoothing using CRF by execute:

	./examples/tangseng/convert_h5_to_png.sh

7. Compile CRF by execute:

	make -C examples/tangseng/crf

8. Run CRF smoothing by execute:

	./examples/tangseng/run_crf.sh

9. Run CRF evaluation by execute:

	./examples/tangseng/crf_eval.sh	

10. Create symbolic links to output images and refined output images of networks by execute:

	./examples/tangseng/createLinkScript.sh

	The links are in public/fashionpose/ along with evaluation result in json format. Json files can be open using following command:

		python -m json.tool <json_file> | less

Miscellaneous
-------------------------------
I have uploaded my utility library as myutil.py. It contains functions for deprocess an image in h5 files to a regular image for plot, show segmentation maps with colors, etc. 