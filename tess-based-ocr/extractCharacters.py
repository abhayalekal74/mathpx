import os
import cv2
import sys
import numpy as np

minH = 2
minW = 2

imgPath = sys.argv[1]
img = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
img = cv2.GaussianBlur(img, (3,3), 0)
img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 75, 10)
img = cv2.bitwise_not(img)
ret, thresh = cv2.threshold(img, 127, 255, 0)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

i = 0
for contour in contours:
	x, y, w, h = cv2.boundingRect(contour)
	if w < minW or h < minH:
		continue
	print (x, y, w, h)
	roi = img[y: y + h, x: x + w]
	cv2.imwrite(os.path.join('extractedImages', str(i) + '.jpeg'), roi)
	i += 1
