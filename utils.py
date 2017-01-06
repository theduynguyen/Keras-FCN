import keras.backend as K
import tensorflow as tf

import pandas as pd
import time


def config_tf():
	# reduce TF verbosity
	tf.logging.set_verbosity(tf.logging.FATAL)

	# prevent from allocating all memory
	config = tf.ConfigProto()
	config.gpu_options.allow_growth=True
	config.allow_soft_placement=True
	
	sess = tf.Session(config=config)
	K.set_session(sess)

def get_exp_id(exp_id_file) :
	exp_csv = pd.read_csv(exp_id_file)
	today = time.strftime('%Y%m%d')
	idx = -1

	# already experimented today?
	try:
	    idx = exp_csv['Timestamp'].astype(str).tolist().index(today)
	except ValueError:
	    pass

	# assign experiment ID
	exp_id = 0
	if idx < 0:
		exp_csv = exp_csv.append({'Timestamp' : today, 'NExp' : 0},ignore_index=True)
	else:
		exp_id = exp_csv.ix[idx,'NExp']
		exp_csv.ix[idx,'NExp'] = exp_id + 1

	exp_csv.to_csv(exp_id_file,index=False)

	return today, exp_id