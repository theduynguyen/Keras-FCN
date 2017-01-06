import numpy as np
import os

from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input

import skimage.transform
import skimage.color
import skimage.io

def pad_image(img,img_size) :
	max_dim = np.argmax(img.shape)
	min_dim = 1 - max_dim

	#resize the largest dim to img_size
	#if img.shape[max_dim] >= img_size:
	resize_factor = np.float(img_size) / np.float(img.shape[max_dim])
	new_min_dim_size = np.round( resize_factor * np.float(img.shape[min_dim]) )
	new_size = [img_size,img_size,3]
	new_size[min_dim] = new_min_dim_size

	img = skimage.transform.resize(np.uint8(img),new_size) 

	# pad dims
	pad_max = img_size - img.shape[max_dim]
	pad_min = img_size - img.shape[min_dim]

	pad = [[0,0],[0,0]]
	pad[max_dim][0] = np.int(np.round(pad_max / 2.0))
	pad[max_dim][1] = np.int(pad_max - pad[max_dim][0])

	pad[min_dim][0] = np.int(np.round(pad_min / 2.0))
	pad[min_dim][1] = np.int(pad_min - pad[min_dim][0])

	pad_tuple = ( (pad[0][0],pad[0][1]), (pad[1][0],pad[1][1]), (0,0))
	img = np.pad(img,pad_tuple,mode='constant')

	return img

def seg_data_generator(stride,n_classes,img_dir,label_dir,img_list,preprocess = True):
	while 1:
		LUT = np.eye(n_classes)

		for img_id in img_list:

			# load image
			img_path = img_dir + img_id
			x = skimage.io.imread(img_path)

			# load label
			label_path = label_dir + img_id[:-3] + 'png'
			y = skimage.io.imread(label_path) # interprets the image as a colour image
			
			#only yield is the images exist
			is_img = type(x) is np.ndarray and type(y) is np.ndarray
			not_empty = len(x.shape) > 0 and len(y.shape) > 0 

			if  is_img and not_empty:
				#deal with gray value images
				if len(x.shape) == 2:
					x = skimage.color.gray2rgb(x)

				# only take one channel
				if len(y.shape) > 2:
					y = y[...,0] 

				# treat binary images
				if np.max(y) == 255:
					y = np.clip(y,0,1)

				# crop if image dims do not match stride
				w_rest = x.shape[0] % stride
				h_rest = x.shape[1] % stride
				
				if w_rest > 0:
					w_crop_1 = np.round(w_rest / 2)
					w_crop_2 = w_rest - w_crop_1
					
					x = x[w_crop_1:-w_crop_2,:,:]
					y = y[w_crop_1:-w_crop_2,:]
				if h_rest > 0:
					h_crop_1 = np.round(h_rest / 2)
					h_crop_2 = h_rest - h_crop_1

					x = x[:,h_crop_1:-h_crop_2,:]
					y = y[:,h_crop_1:-h_crop_2]

				# prepare for NN
				x = np.array(x,dtype='float')
				x = x[np.newaxis,...]

				if preprocess == True:
					x = preprocess_input(x)

				y = LUT[y]
				y = y[np.newaxis,...] # make it a 4D tensor

				yield x, y

import scipy.io as sio
import skimage.draw

def pose_data_generator(stride,n_joints,img_dir,label_file,img_list,preprocess = True):
	while 1:
		# load joint annotation file
		annot_struct = sio.loadmat(label_file)
		annot = annot_struct['joints'] # order (dim,joint_id,image_no)
		
		resize_factor = 4
		annot *= resize_factor

		for img_id in img_list:

			# load image
			img_path = img_dir + 'im' + format(img_id, '04') + '.jpg'
			x = skimage.io.imread(img_path)
			x = skimage.transform.resize(x,
										(x.shape[0]*resize_factor,x.shape[1]*resize_factor))

			# load label
			joint_coords = annot[:,:,img_id-1]
			
			#only yield is the images exist
			is_img = type(x) is np.ndarray
			not_empty = len(x.shape) > 0

			if  is_img and not_empty:
				#deal with gray value images
				if len(x.shape) == 2:
					x = skimage.color.gray2rgb(x)

				# crop if image dims do not match stride
				w_rest = x.shape[0] % stride
				h_rest = x.shape[1] % stride
				
				if w_rest > 0:
					w_crop_1 = np.round(w_rest / 2)
					w_crop_2 = w_rest - w_crop_1
					
					x = x[w_crop_1:-w_crop_2,:,:]
				if h_rest > 0:
					h_crop_1 = np.round(h_rest / 2)
					h_crop_2 = h_rest - h_crop_1

					x = x[:,h_crop_1:-h_crop_2,:]

				# create label volume
				y = np.zeros((x.shape[0],x.shape[1],n_joints),dtype='uint8')
				radius = 0.05*x.shape[0]	

				for j in range(n_joints):
					xmin = max( int(joint_coords[0,j] - radius), 0 )
					xmax = min( int(joint_coords[0,j] + radius), x.shape[1] )
					ymin = max( int(joint_coords[1,j] - radius), 0 )
					ymax = min( int(joint_coords[1,j] + radius), x.shape[0] )

					y[ymin:ymax,xmin:xmax,j] = 1.0

				# prepare for NN
				x = np.array(x,dtype='float')
				x = x[np.newaxis,...]

				if preprocess == True:
					x = preprocess_input(x)

				y = y[np.newaxis,...] # make it a 4D tensor

				yield x, y