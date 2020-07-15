import numpy as np
import cv2
import os
import glob
from random import shuffle

class Dataset():

	def __init__(self, path, shuffle=False):

		self.x, self.y, self.image_paths = [], [], []
		image_paths = os.path.join(path, '*.jpg')
		self.image_paths = glob.glob(image_paths)
		if shuffle:
			shuffle(self.image_paths)
		
	def getData(self):
		return self.image_paths

	def getBatch(self, batch_idx, batch_size):
		start = batch_idx*batch_size
		end = batch_size*(batch_idx+1)
		if end > len(self.image_paths):
			end = len(self.image_paths)
		images = np.array([cv2.imread(image_path) for image_path in self.image_paths[start:end]])/255.
		x = images[:, :, :256, :]
		y = images[:, :, 256:, :]
		return x, y