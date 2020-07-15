import matplotlib.pyplot as plt

def plotFig(img, use_path=True, debug=False, axis='on'):
	'''
		Plot image from a given image path/image
	'''
	if use_path:
		img = plt.imread(img)
	fig = plt.figure(figsize=(8, 8))
	plt.imshow(img)
	plt.axis(axis)
	if debug:
		plt.show()
	else:
		return fig

