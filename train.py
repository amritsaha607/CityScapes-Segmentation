import os
import glob
import yaml
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# from data.dataset import Dataset
from data.generator import DataGenerator
from models.unet import UNet
from models.callback import CustomCallback
from utils.processwandb import processWandb

random.seed(0)
np.random.seed(0)
tf.random.set_seed(0)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--version', type=str, default='v1', help='version of the experiment')

args = parser.parse_args()
config_path = 'config/{}.yml'.format(args.version)
all_configs = yaml.safe_load(open(config_path))
all_configs['run_name'] = 'train_{}'.format(args.version)
N_EPOCHS = all_configs['n_epochs']
batch_size = all_configs['batch_size']
data_root = all_configs['data_root']
train_annot, val_annot = all_configs['train_annot'], all_configs['val_annot']
H, W = all_configs['H'], all_configs['W']

ckpt_dir = os.path.join(all_configs['checkpoints'], args.version)
if not os.path.exists(all_configs['checkpoints']):
	os.makedirs(all_configs['checkpoints'])
all_configs['ckpt_dir'] = ckpt_dir

processWandb(all_configs)

def bakeGenerator(annot):
	lines = open(annot, 'r').read().strip().split('\n')
	x_set = [line.split()[0] for line in lines]
	y_set = [line.split()[1] for line in lines]
	# x_set = glob.glob(os.path.join(root, 'raw', '*.jpg'))
	# y_set = glob.glob(os.path.join(root, 'mask', '*.jpg'))
	gen = DataGenerator(x_set, y_set, batch_size=batch_size, shuffle=True)
	return gen

train_gen = bakeGenerator(train_annot)
val_gen = bakeGenerator(val_annot)

model = UNet()

hist = model.fit(
	train_gen,
	validation_data=val_gen,
	epochs=N_EPOCHS,
	batch_size=batch_size,
	verbose=1,
	use_multiprocessing=True,
	workers=8,
	callbacks=[CustomCallback(ckpt_dir),],
)

