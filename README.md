# Keras-FCN
This repository contains my implementation of [Fully Convolutional Networks](http://fcn.berkeleyvision.org) in Keras (Tensorflow backend). Currently, semantic segmentation can be performed. In contrast to the original implementation with Caffe, I have used the following modifications:

- Instead of a VGG feature extractor, I use a Resnet50 feature extractor.
- The deconvolution layer has been realised by an upsampling followed by a 1x1 convolution. I have not found a way, yet, to use the Deconvolution2D operation in Keras with flexible sized images. If somebody knows a solution, feel free to contact me.

## Required packages

- Tensorflow
- Keras
- Pandas
- Matplotlib (for result visualisation)
- Scikit Image

## Usage
For training, use `train_segmentation.py -d Your_Image_Folder` script. The image folder has to contain the directories 'train_img', 'train_labels', 'val_img', 'val_labels' and a 'labels.txt' file which contains all the labels separated by a newline. The label format are expected to be a one channel image containing the label for each pixel. During training, the script saves the the weights achieving the best validation loss into the destination directory.

To visualise the results, use `debug_predict.py -mi Your_Path_to_model`. 
