import numpy as np
import tensorflow as tf

def accuracy(y, y_pred):
	n = np.prod(y.shape)
	n_match = (y==y_pred).sum()
	return n_match/n

def acc(y, y_pred):
	'''
		y & y_pred are tensors (float)
	'''

	y = tf.cast(y*255, dtype=tf.int32)
	y_pred = tf.cast(y_pred*255, dtype=tf.int32)
	y = tf.keras.backend.flatten(y)
	y_pred= tf.keras.backend.flatten(y_pred)
	n = y.shape[0]
	n_mismatch = tf.math.count_nonzero(y-y_pred)
	return (n-n_mismatch)/n

