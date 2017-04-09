import matplotlib.pyplot as plt
import matplotlib.patches as patches

import skimage.io as io
import shutil
import pandas as pd
import numpy as np
import collections

coco_dir = '/home/tdnguyen/data/COCO/coco/'
out_dir = coco_dir + 'Segmentation/'
annot_dir = coco_dir + 'annotations/'

import sys
sys.path.append(coco_dir + 'PythonAPI/pycocotools')
sys.path.append(coco_dir + 'PythonAPI')
import pycocotools
from pycocotools.coco import COCO
from pycocotools.coco import mask

train_annotation_file = annot_dir + 'instances_train2014.json'
train_src_img_dir = coco_dir + 'train2014/'
train_img_dir = out_dir + 'train_images/'
train_label_dir = out_dir + 'train_labels/'

val_annotation_file = annot_dir + 'instances_val2014.json'
val_src_img_dir = coco_dir + 'val2014/'
val_img_dir = out_dir + 'val_images/'
val_label_dir = out_dir + 'val_labels/'

label_file = out_dir + 'labels.txt'

#Create output dirs if they do not exist
import os
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

if not os.path.exists(train_img_dir):
    os.makedirs(train_img_dir)
if not os.path.exists(train_label_dir):
    os.makedirs(train_label_dir)

if not os.path.exists(val_img_dir):
    os.makedirs(val_img_dir)
if not os.path.exists(val_label_dir):
    os.makedirs(val_label_dir)

################## Params ##########################
# train or val set?
train = False

# how many input images?
n_samples = 200

min_img_size = 0
min_bbox_size = 0

# which labels shall the images include?
seg_labels = ['person','backpack','laptop','handbag','suitcase']
#LUT = np.array([[0,0,0],[255,0,0],[0,255,0],[0,0,255]])
LUT = np.array(range(len(seg_labels)+1),dtype='uint8')

annotation_file = val_annotation_file
src_img_dir = val_src_img_dir
img_dir = val_img_dir
label_dir = val_label_dir

if train == True:
	annotation_file = train_annotation_file
	src_img_dir = train_src_img_dir
	img_dir = train_img_dir
	label_dir = train_label_dir

####################################################
img_sample_idx = range(n_samples)
coco = COCO(annotation_file)

# create label - id dictionary
cat_ids = coco.getCatIds(seg_labels)
cat_rec = coco.loadCats(cat_ids)
label_dict = collections.OrderedDict()

label_count = 1
for c in cat_rec:
	label_dict[c['id']] = (c['name'],label_count)
	label_count += 1

print label_dict

# get images containing objects
img_ids = coco.getImgIds()

img_ids_sampled = [img_ids[i] for i in img_sample_idx]
img_recs = coco.loadImgs(img_ids_sampled)

######################################
record_id = 0

for record in img_recs:
	img_filename = record['file_name']
	
	dest_filename = format(record_id, '06')
	#dest_filename = img_filename[:-3]
	
	dest_img_suffix = img_filename[-4:]
	size_ok = True	
	gray_img = False

	# load image and convert into RGB
	src_file = src_img_dir + img_filename
	dest_file = img_dir + dest_filename + dest_img_suffix 
	img = io.imread(src_file)

	#if len(img.shape) < 3:
	#	img = io.gray2rgb(img)
	#	gray_img = True
	
	#no small images
	if img.shape[0] < min_img_size or img.shape[1] < min_img_size:
		create_dat_point = False
		continue

	
	# create annotation file
	ann_ids = coco.getAnnIds(imgIds = record['id'])
	annot_rec = coco.loadAnns(ann_ids)
	
	# add annotation polygons
	annot_list = []
	contains_label = False
	crowd_img = False
	for a in annot_rec:
		label = a['category_id']

		if label in label_dict.keys():
			if len(a['segmentation']) == 1:
				RLE = mask.frPyObjects(a['segmentation'], img.shape[0], img.shape[1])
				annot_list.append((RLE,label_dict[label][1]))
			else:
				crowd_img = True

			contains_label = True
	

	# create actual data point
	if size_ok == True and contains_label and not crowd_img:
		record_id += 1

		# copy image into destination folder
		# shutil.copyfile(src_file,dest_file)
		io.imsave(dest_file,img)

		m = np.zeros((img.shape[0],img.shape[1],1),dtype=np.uint8)
		for a in annot_list:
			sub_mask = mask.decode(a[0])
			sub_mask[sub_mask > 0] = a[1]

			m = np.fmax(m,sub_mask)
		
		#convert images into color images
		annot_filename = label_dir + dest_filename + '.png'
		
		#label_img = LUT[m]
		label_img = m.astype('uint8')
		io.imsave(annot_filename,label_img)

#save label file
file = open(label_file, "w")
file.write('background\n')
for l in seg_labels:
	file.write(l+'\n')
