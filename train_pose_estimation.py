import numpy as np
import random
import os
import argparse

import pandas as pd
import tensorflow as tf
import keras.backend as K
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras.optimizers import Adam, SGD

from model_fcn import resnet50_fcn, testnet_fcn, resnet50_16s_fcn, resnet50_8s_fcn
from data_generator import pose_data_generator, pad_image
import utils

def parse_args():
	parser = argparse.ArgumentParser()

	parser.add_argument('-t', '--n_train_img', help='Number of train images', 
						type=int,default=1000)

	parser.add_argument('-v', '--n_val_img', help='Number of validation images', 
						type=int,default=100)

	parser.add_argument('-n', '--net', help='Net to train (testnet, resnet50 or resnet50_16s)', 
						default='resnet50')

	parser.add_argument('-e', '--epochs', help='Number of epochs', 
						type=int, default=1)

	parser.add_argument('-g', '--gpu', help='Use GPU ID', 
						type=int, default=0)

	parser.add_argument('-o', '--opt', help='Optimizer', 
						default='SGD')

	parser.add_argument('-d', '--img_dir', help='Directory containing the images', 
						default='/data/lsp_dataset/images/')

	parser.add_argument('-lr', '--learning_rate', help='Initial learning rate', 
						default=0.01)

	parser.add_argument('-mi', '--model_input', help='Init with model', 
					default='')

	parser.add_argument('-mo', '--model_output', help='Where to save the trained model?', 
					default='/work/fcn_models/')

	parser.add_argument('-id', '--exp_id', help='Experiment id', 
					default='')

	return parser.parse_args()


######################################################################################

args = parse_args()
utils.config_tf()
exp_id_file = args.model_output + 'Exp_ID.csv'

# create experimental directory
if args.exp_id == '':
	# use date + number of exps so far today
	today, exp_id = utils.get_exp_id(exp_id_file)
	model_output_dir = args.model_output + str(today) + '_' + str(exp_id).zfill(2)
else :
	model_output_dir = args.model_output + args.exp_id

if not os.path.exists(model_output_dir):
    os.makedirs(model_output_dir)


# set vars
N_train_img = args.n_train_img
N_val_img = args.n_val_img
N_epochs = args.epochs
n_joints = 14
	
# create model
gpu = '/gpu:' + str(args.gpu)
with tf.device(gpu):
	# create model
	if args.net == 'resnet50' :
		model, stride = resnet50_fcn(n_joints)

	if args.net == 'resnet50_16s' :
		model, stride = resnet50_16s_fcn(n_joints,args.model_input)

	if args.net == 'resnet50_8s' :
		model, stride = resnet50_8s_fcn(n_joints,args.model_input)

	if args.net == 'testnet' :
		model, stride = testnet_fcn(n_joints)

# create data generators
img_dir = '/data/lsp_dataset/images/'
label_file = '/data/lsp_dataset/joints.mat'
img_list = range(1,2000)

train_gen = pose_data_generator(stride,n_joints,img_dir,label_file,img_list[:N_train_img])
val_gen = pose_data_generator(stride,n_joints,img_dir,label_file,img_list[N_train_img:N_train_img+N_val_img])

# callbacks
filepath = model_output_dir + '/best_weights.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, 
							save_best_only=True, mode='min')

tb = TensorBoard(log_dir = model_output_dir, histogram_freq= 2, write_graph=False)

plateau = ReduceLROnPlateau(patience=5)

callbacks_list = [checkpoint, tb, plateau]


learning_rate = float(args.learning_rate)
if args.opt == 'Adam':
	opt = Adam(lr=learning_rate)
elif args.opt == 'SGD':
	opt = SGD(lr=learning_rate, momentum=0.9)

model.compile(optimizer = opt,loss = 'mse')

print model.summary()

model.fit_generator(train_gen,
					samples_per_epoch=N_train_img, nb_epoch=N_epochs,
					validation_data = val_gen, nb_val_samples = N_val_img,
					callbacks=callbacks_list,verbose=1)