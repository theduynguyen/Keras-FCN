import numpy as np
import random
import os
import argparse
import matplotlib.pyplot as plt

import tensorflow as tf
import keras.backend as K
from keras.models import Model

from model_fcn import resnet50_fcn, resnet50_16s_fcn, resnet50_8s_fcn
from data_generator import pose_data_generator, pad_image
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

n_joints = 14
n_rows = 5
n_cols = 4

# create model
gpu = '/gpu:' + str(args.gpu)
with tf.device(gpu):
	if args.net == 'resnet50' : 
		model, stride = resnet50_fcn(n_joints)

	if args.net == 'resnet50_16s' :
		model, stride = resnet50_16s_fcn(n_joints)

	if args.net == 'resnet50_8s' :
		model, stride = resnet50_8s_fcn(n_joints) 

	if args.model_input_dir != '':
		model.load_weights(args.model_input_dir + '/best_weights.hdf5')

# load image and predict
N_train_img = 1000
img_dir = '/home/tdnguyen/data/lsp_dataset/images/'
label_file = '/home/tdnguyen/data/lsp_dataset/joints.mat'
img_list = range(1,2000)

pred_gen = pose_data_generator(stride,n_joints,img_dir,label_file,
							  img_list[N_train_img+args.img_id:])
val_gen = pose_data_generator(stride,n_joints,img_dir,label_file,
							  img_list[N_train_img+args.img_id:],
							  preprocess=False)

# Visualise in tiled plot
fig = plt.figure()

for idx in range(n_rows) :
	for row_img in range(2) :
		x_img, _ = val_gen.next()
		
		x, y = pred_gen.next()
		y = np.argmax(y,axis =-1)[0]

		pred = model.predict(x,batch_size=1)[0]
		
		# Visualise result
		fig.add_subplot(n_rows,6,idx*6 + row_img*3 +1)
		plt.imshow(x_img[0])
		
		fig.add_subplot(n_rows,6,idx*6 + row_img*3 + 2)
		plt.imshow(y, vmin=0,vmax=n_joints-1)
			
		fig.add_subplot(n_rows,6,idx*6 + row_img*3 + 3)
		plt.imshow(pred[:,:,0], vmin=0,vmax=1)

plt.show()