import keras
import time
import pickle
import json
import os
from scipy.misc import imresize
import cv2
import sys
import numpy as np
from mergeChars import getLatex

minH = 10 
minW = 10
padding = 10
modelShape = 50


class Rectangle():
	def __init__(self, l, t, r, b):
		self.left = l
		self.top = t
		self.right = r
		self.bottom = b
		self.prediction = None
		self.paddedPixelMatrix = None


	def print(self):
		print (self.prediction, self.left, self.top, self.right, self.bottom)


def getSkew(img):
	# grab the (x, y) coordinates of all pixel values that
	# are greater than zero, then use these coordinates to
	# compute a rotated bounding box that contains all
	# coordinates
	coords = np.column_stack(np.where(img == 0))
	angle = cv2.minAreaRect(coords)[-1]

	# the `cv2.minAreaRect` function returns values in the
	# range [-90, 0); as the rectangle rotates clockwise the
	# returned angle trends to 0 -- in this special case we
	# need to add 90 degrees to the angle
	if angle < -45:
		angle = -(90 + angle)

	# otherwise, just take the inverse of the angle to make
	# it positive
	else:
		angle = -angle
	return angle


def deskew(img, skewAngle):
	# rotate the image to deskew it
	(h, w) = img.shape[:2]
	center = (w // 2, h // 2)
	M = cv2.getRotationMatrix2D(center, skewAngle, 1.0)
	rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
	cv2.imwrite(os.path.join('modifiedImages', 'rotated.jpeg'), rotated)
	return rotated


def getContours(imgPath):
	img = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
	img = cv2.GaussianBlur(img, (3,3), 0)
	img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 475, 10)
	#img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
	skewAngle = getSkew(img)
	print ("\nskewAngle", skewAngle)
	#img = deskew(img, skewAngle)
	contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	return img, contours


def getBoundingRects(img, contours):
	rectangles = list()
	imgCopy = np.copy(img)
	for contour in contours:
		x, y, w, h = cv2.boundingRect(contour)
		if w < minW and h < minH:
			continue
		r = Rectangle(x, y, x + w, y + h) 
		r.paddedPixelMatrix = np.pad(img[y: y + h, x: x + w], (padding, padding), 'constant', constant_values=(255, 255)) 
		rectangles.append(r)
		cv2.rectangle(imgCopy, (x, y), (x + w, y + h), (150, 150, 150), 2)
	cv2.imwrite(os.path.join('modifiedImages', 'bounds.jpeg'), imgCopy)
	return rectangles
	

def predict(model, rectangles):	
	labelEncoder = pickle.load(open('char-recognition-model/labelEncoder.pkl', 'rb'))
	with open('char-recognition-model/classcodes.json', 'r') as f:
		classCodes = json.load(f)
	classArgmax = {}
	for c in labelEncoder.classes_:
		classArgmax[np.argmax(labelEncoder.transform([[c,]]))] = c
	imgs = np.array([imresize(r.paddedPixelMatrix, (modelShape, modelShape)) for r in rectangles])
	res = model.predict(imgs)
	for i in range(len(res)):
		pred = classCodes[classArgmax[np.argmax(res[i])]]
		if "rightarrow" in pred or "leftarrow" in pred:
			pred = "frac"
		if ' ' in pred:
			#pred = ' {\sym ' + pred + '} '
			pred = ''
		rectangles[i].prediction = pred 

if __name__=='__main__':
	s1 = time.time()
	model = keras.models.load_model(sys.argv[1])
	s2 = time.time()
	img, contours = getContours(sys.argv[2])
	s3 = time.time()
	rectangles = getBoundingRects(img, contours)
	s4 = time.time()
	predict(model, rectangles)
	s5 = time.time()
	getLatex(rectangles)
	s6 = time.time()
	print ("Total time", (s6 - s1) * 1000)
	"""
	print ("Model load", (s2 - s1) * 1000)
	print ("getContours", (s3 - s2) * 1000)
	print ("getBoundingRects", (s4 - s3) * 1000)
	print ("predict", (s5 - s4) * 1000)
	print ("get latex", (s6 - s5) * 1000)
	"""
