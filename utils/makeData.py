import glob
import os
import cv2
from tqdm import tqdm

def build(src, out, rm=True):
	raw_dir = os.path.join(out, 'raw')
	mask_dir = os.path.join(out, 'mask')
	if not os.path.exists(raw_dir):
		os.makedirs(raw_dir)
	if not os.path.exists(mask_dir):
		os.makedirs(mask_dir)
	files = glob.glob(os.path.join(src, '*.jpg'))
	for file in tqdm(files):
		img = cv2.imread(file)
		f_name = file.split('/')[-1]
		raw_path = os.path.join(raw_dir, f_name)
		mask_path = os.path.join(mask_dir, f_name)
		raw, mask = img[:, :256, :], img[:, 256:, :]
		cv2.imwrite(raw_path, raw)
		cv2.imwrite(mask_path, mask)
	if rm:
		os.system('rm {}'.format(str(os.path.join(src, '*jpg'))))

build(src='/content/data/cityscapes_data/train', out='/content/data/cityscapes_data/train')
build(src='/content/data/cityscapes_data/val', out='/content/data/cityscapes_data/val')