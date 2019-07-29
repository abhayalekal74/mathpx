import os
import sys

remFiles = []
for root, dirs, files in os.walk(sys.argv[1]):
	remFiles += [os.path.join(root, f) for f in files if f.endswith('.jpeg') and ('mod' in f)]

for f in remFiles:
	print (f)
	os.remove(f)
