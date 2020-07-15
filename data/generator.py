import tensorflow as tf
import numpy as np
import random
import cv2

class DataGenerator(tf.keras.utils.Sequence):

    def __init__(self, raw_files, mask_files, batch_size=4, shuffle=True):
        # 'Initialization'
        self.raw_files = raw_files
        self.mask_files = mask_files
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.raw_files) / self.batch_size))

    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        raw_files_batch = [self.raw_files[k] for k in indexes]
        mask_files_batch = [self.mask_files[k] for k in indexes]

        # Generate data
        x = self.__data_generation(raw_files_batch)
        y = self.__data_generation(mask_files_batch)

        return x, y

    def on_epoch_end(self):
        # 'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.raw_files))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


    def __data_generation(self, files):
        imgs = []

        for file in files:

            img = cv2.imread(file)/255.

            ###############
            # Augment image
            ###############

            imgs.append(img) 

        return np.array(imgs)