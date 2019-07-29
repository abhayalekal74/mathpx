from scipy.misc import imread, imsave
import os
import sys
import random
import numpy as np


minGreyPix = 100
maxGreyPix = 150
noisyImageCopies = 5 
minNoisePerc = 10
maxNoisePerc = 30

for root, dirs, files in os.walk(sys.argv[1]):
	for f in files:
		if f.endswith('.jpeg') and 'mod' not in f:
			print (os.path.join(root, f))
			for n in range(noisyImageCopies):
				fPath = os.path.join(root, f)
				img = imread(fPath, mode='L')
				totalPix = img.shape[0] * img.shape[1]
				pixelsToBeGreyed = int(totalPix * (random.randrange(minNoisePerc, maxNoisePerc) / 100.0))
				for i in range(pixelsToBeGreyed):
					greyPixPos = random.randrange(totalPix)
					row = greyPixPos // img.shape[1]
					col = greyPixPos % img.shape[1]
					randomGreyPix = random.choice(np.arange(minGreyPix, maxGreyPix))
					img[row][col] = randomGreyPix
				imsave(fPath[:fPath.rindex('.')] + '-mod-' + str(n) + '.jpeg', img)
			
