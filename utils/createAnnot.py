import glob
import os

def bake(src, out, max_=None):
	raws = sorted(glob.glob(os.path.join(src, 'raw', '*.jpg')))
	masks = sorted(glob.glob(os.path.join(src, 'mask','*.jpg')))
	if max_ is not None:
		raws = raws[:max_]
		masks = masks[:max_]
	f = open(out, 'w')
	for idx in range(len(raws)):
		f.write('{} {}\n'.format(raws[idx], masks[idx]))
	f.close()

# bake(
# 	src='/content/data/cityscapes_data/train/',
# 	out='../assets/train.txt',
# )
# bake(
# 	src='/content/data/cityscapes_data/val/',
# 	out='../assets/val.txt',
# )
# bake(
# 	src='/content/data/cityscapes_data/train/',
# 	out='../assets/train_100.txt',
# 	max_=100,
# )
# bake(
# 	src='/content/data/cityscapes_data/val/',
# 	out='../assets/val_100.txt',
# 	max_=100,
# )