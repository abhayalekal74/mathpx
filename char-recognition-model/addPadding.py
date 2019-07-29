import os
import sys
from scipy.misc import imread, imsave
import numpy as np


pad_width = 10

for r, d, fls in os.walk(sys.argv[1]):
	for fl in fls:
		img = imread(os.path.join(r, fl), 'L')
		#img = np.pad(img, pad_width, mode='constant', constant_values=(255))
		for i in range(10):
			for j in range(img.shape[1]):
				img[i][j] = 255
				img[img.shape[0] - i - 1][j] = 255
		for i in range(10, img.shape[0] - 9):
			for j in range(10):
				img[i][j] = 255
				img[i][img.shape[1] - j - 1] = 255
		print (fl)
		imsave(os.path.join(r, fl), img)
