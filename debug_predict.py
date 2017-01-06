import numpy as np
import random
import os
import argparse
import matplotlib.pyplot as plt

import tensorflow as tf
import keras.backend as K
from keras.models import Model

from model_fcn import resnet50_fcn, resnet50_16s_fcn, resnet50_8s_fcn
from data_generator import seg_data_generator, pad_image
import utils

def parse_args():
	parser = argparse.ArgumentParser()
	
	parser.add_argument('-d', '--img_dir', help='Directory containing the images', 
						default='/home/tdnguyen/data/COCO/data/Segmentation/')

	parser.add_argument('-i', '--img_id', help='Image ID', 
					type=int, default=0)

	parser.add_argument('-n', '--net', help='Net (resnet50 or resnet50_16s)', default='resnet50')


	parser.add_argument('-mi', '--model_input_dir', help='Where the model is saved')

	parser.add_argument('-g', '--gpu', help='Use GPU ID', 
						type=int, default=0)

	return parser.parse_args()

args = parse_args()
utils.config_tf()

n_classes = 6
n_rows = 10

# create model
gpu = '/gpu:' + str(args.gpu)
with tf.device(gpu):
	if args.net == 'resnet50' : 
		model, stride = resnet50_fcn(n_classes)

	if args.net == 'resnet50_16s' :
		model, stride = resnet50_16s_fcn(n_classes)

	if args.net == 'resnet50_8s' :
		model, stride = resnet50_8s_fcn(n_classes) 

	if args.model_input_dir != '':
		model.load_weights(args.model_input_dir + '/best_weights.hdf5')

# load image and predict
val_img_dir = args.img_dir + 'val_images/'
val_label_dir = args.img_dir + 'val_labels/'

img_list_val = os.listdir(val_img_dir)
img_list_val = img_list_val[args.img_id:]
pred_gen = seg_data_generator(stride,n_classes,val_img_dir,val_label_dir,img_list_val)
val_gen = seg_data_generator(stride,n_classes,val_img_dir,val_label_dir,img_list_val,preprocess = False)

# Visualise in tiled plot
fig = plt.figure()

for idx in range(n_rows) :
	for row_img in range(2) :
		x_img, _ = val_gen.next()
		x_img = np.array(x_img,dtype='uint8')

		x, y = pred_gen.next()
		y = np.argmax(y,axis =-1)[0]

		pred = model.predict(x,batch_size=1)[0]
		pred = np.argmax(pred,axis =-1)

		# Visualise result
		fig.add_subplot(n_rows,6,idx*6 + row_img*3 +1)
		plt.imshow(x_img[0])
		
		fig.add_subplot(n_rows,6,idx*6 + row_img*3 + 2)
		plt.imshow(y, vmin=0,vmax=n_classes-1)
		
		fig.add_subplot(n_rows,6,idx*6 + row_img*3 + 3)
		plt.imshow(pred, vmin=0,vmax=n_classes-1)

plt.show()