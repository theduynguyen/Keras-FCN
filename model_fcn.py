import keras.backend as K
from keras.engine import Layer

from keras.layers import Input, Dropout, merge
from keras.layers.convolutional import Convolution2D, UpSampling2D, ZeroPadding2D, Cropping2D, Deconvolution2D
from keras.layers.core import Activation

from keras.applications.resnet50 import ResNet50
from keras.models import Model, load_model

import numpy as np

class Softmax4D(Layer):
    def __init__(self, axis=-1,**kwargs):
        self.axis=axis
        super(Softmax4D, self).__init__(**kwargs)

    def build(self,input_shape):
        pass

    def call(self, x,mask=None):
        e = K.exp(x - K.max(x, axis=self.axis, keepdims=True))
        s = K.sum(e, axis=self.axis, keepdims=True)
        return e / s

    def get_output_shape_for(self, input_shape):
		return input_shape

def bilinear_interpolation(w):
	frac = w[0].shape[0]
	n_classes = w[0].shape[-1]
	w_bilinear = np.zeros(w[0].shape)

	for i in range(n_classes):
		w_bilinear[:,:,i,i] = 1.0/(frac*frac) * np.ones((frac,frac))

	return w_bilinear

def resnet50_fcn(n_classes):
	# load ResNet
	input_tensor = Input(shape=(None, None, 3))
	base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=input_tensor)

	# add classifier
	x = base_model.get_layer('activation_49').output
	x = Dropout(0.5)(x)
	x = Convolution2D(n_classes,1,1,name = 'pred_32',init='zero',border_mode = 'valid')(x)
	
	# add upsampler
	stride = 32
	x = UpSampling2D(size=(stride,stride))(x)
	x = Convolution2D(n_classes,5,5,name = 'pred_32s',init='zero',border_mode = 'same')(x)
	x = Softmax4D(axis=-1)(x)
	
	model = Model(input=base_model.input,output=x)

	# create bilinear interpolation
	w = model.get_layer('pred_32s').get_weights()
	model.get_layer('pred_32s').set_weights([bilinear_interpolation(w), w[1]])
	
	# fine-tune 
	train_layers = ['pred_32',
					'pred_32s'
					
					'bn5c_branch2c', 
					'res5c_branch2c',
					'bn5c_branch2b', 
					'res5c_branch2b',
					'bn5c_branch2a', 
					'res5c_branch2a',

					'bn5b_branch2c', 
					'res5b_branch2c',
					'bn5b_branch2b', 
					'res5b_branch2b',
					'bn5b_branch2a', 
					'res5b_branch2a',
					
					'bn5a_branch2c', 
					'res5a_branch2c',
					'bn5a_branch2b', 
					'res5a_branch2b',
					'bn5a_branch2a', 
					'res5a_branch2a']

	for l in model.layers:
		if l.name in train_layers:
			l.trainable = True
		else :
			l.trainable = False

	return model, stride

def resnet50_16s_fcn(n_classes,model_input = ''):
	# load 32s base model
	base_model, stride = resnet50_fcn(n_classes)

	if model_input != '':
		base_model.load_weights(model_input)
	
	# add 16s classifier
	x = base_model.get_layer('activation_40').output
	x = Dropout(0.5)(x)
	x = Convolution2D(n_classes,1,1,name = 'pred_16',init='zero',border_mode = 'valid')(x)
	x = UpSampling2D(name='upsampling_16',size=(stride/2,stride/2))(x)
	x = Convolution2D(n_classes,5,5,name = 'pred_up_16',init='zero',border_mode = 'same')(x)
	
	# merge classifiers
	x = merge([x, base_model.get_layer('pred_32s').output],mode = 'sum')
	x = Softmax4D(name='pred_16s',axis=-1)(x)
	
	model = Model(input=base_model.input,output=x)

	# create bilinear interpolation
	w = model.get_layer('pred_up_16').get_weights()
	model.get_layer('pred_up_16').set_weights([bilinear_interpolation(w), w[1]])

	# fine-tune 
	train_layers = ['pred_32',
					'pred_32s',
					'pred_16',
					'pred_up_16',
					
					'bn5c_branch2c', 
					'res5c_branch2c',
					'bn5c_branch2b', 
					'res5c_branch2b',
					'bn5c_branch2a', 
					'res5c_branch2a',

					'bn5b_branch2c', 
					'res5b_branch2c',
					'bn5b_branch2b', 
					'res5b_branch2b',
					'bn5b_branch2a', 
					'res5b_branch2a',
					
					'bn5a_branch2c', 
					'res5a_branch2c',
					'bn5a_branch2b', 
					'res5a_branch2b',
					'bn5a_branch2a', 
					'res5a_branch2a']

	for l in model.layers:
		if l.name in train_layers:
			l.trainable = True
		else :
			l.trainable = False

	return model, stride

def resnet50_8s_fcn(n_classes,model_input = ''):
	# load 16s base model
	base_model, stride = resnet50_16s_fcn(n_classes)

	if model_input != '':
		base_model.load_weights(model_input)
	
	# add 16s classifier
	x = base_model.get_layer('activation_22').output
	x = Dropout(0.5)(x)
	x = Convolution2D(n_classes,1,1,name = 'pred_8',init='zero',border_mode = 'valid')(x)
	x = UpSampling2D(name='upsampling_8',size=(stride/4,stride/4))(x)
	x = Convolution2D(n_classes,5,5,name = 'pred_up_8',init='zero',border_mode = 'same')(x)
	
	# merge classifiers
	x = merge([x, base_model.get_layer('pred').output],mode = 'sum')
	x = Softmax4D(name='pred_8s',axis=-1)(x)
	
	model = Model(input=base_model.input,output=x)

	# create bilinear interpolation
	w = model.get_layer('pred_up_8').get_weights()
	model.get_layer('pred_up_8').set_weights([bilinear_interpolation(w), w[1]])

	# fine-tune 
	train_layers = ['pred_32',
					'pred_32s',
					'pred_16',
					'pred_up_16',
					'pred_8',
					'pred_up_8',
					
					'bn5c_branch2c', 
					'res5c_branch2c',
					'bn5c_branch2b', 
					'res5c_branch2b',
					'bn5c_branch2a', 
					'res5c_branch2a',

					'bn5b_branch2c', 
					'res5b_branch2c',
					'bn5b_branch2b', 
					'res5b_branch2b',
					'bn5b_branch2a', 
					'res5b_branch2a',
					
					'bn5a_branch2c', 
					'res5a_branch2c',
					'bn5a_branch2b', 
					'res5a_branch2b',
					'bn5a_branch2a', 
					'res5a_branch2a']

	for l in model.layers:
		if l.name in train_layers:
			l.trainable = True
		else :
			l.trainable = False

	return model, stride

def testnet_fcn(n_classes):
	stride = 32
	input_tensor = Input(shape=(None, None, 3))
	x = Convolution2D(4,5,5,name='conv',
					activation = 'relu', border_mode='same', subsample= (stride,stride))(input_tensor)

	x = Softmax4D(axis=-1)(x)
	x = UpSampling2D(size=(stride,stride))(x)
	x = Convolution2D(n_classes,3,3,name = 'pred_up',border_mode = 'same')(x)
		
	model = Model(input=input_tensor,output=x)
		
	return model, stride