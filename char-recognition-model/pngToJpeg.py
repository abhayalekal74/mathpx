import os
import sys

rootFolder = sys.argv[1]
	
for root, directory, files in os.walk(rootFolder):
	for f in files:
		if f.endswith('.png'):
			fname = os.path.join(root, f)
			os.rename(fname, fname + '.jpeg')
