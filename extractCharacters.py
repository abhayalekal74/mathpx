import keras
import time
import pickle
import json
import os
from scipy.misc import imresize
import cv2
import sys
import numpy as np
from merge_based_on_pixel_pos import get_latex

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


def getContours(imgPath):
	img = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
	img = cv2.GaussianBlur(img, (3,3), 0)
	img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 75, 10)
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
	label_encoder = pickle.load(open('char-recognition-model/label_encoder.pkl', 'rb'))
	with open('char-recognition-model/classcodes.json', 'r') as f:
		class_codes = json.load(f)
	class_argmax = {}
	for c in label_encoder.classes_:
		class_argmax[np.argmax(label_encoder.transform([[c,]]))] = c
	imgs = np.array([imresize(r.paddedPixelMatrix, (modelShape, modelShape)) for r in rectangles])
	res = model.predict(imgs)
	for i in range(len(res)):
		pred = class_codes[class_argmax[np.argmax(res[i])]]
		rectangles[i].prediction = pred if len(pred.split(' ')) == 1 else ''


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
	[r.print() for r in rectangles]
	s6 = time.time()
	get_latex(rectangles)
	s7 = time.time()
	print ("Total time", (s7 - s1) * 1000)
	print ("Model load", (s2 - s1) * 1000)
	print ("getContours", (s3 - s2) * 1000)
	print ("getBoundingRects", (s4 - s3) * 1000)
	print ("predict", (s5 - s4) * 1000)
	print ("print rectangles", (s6 - s5) * 1000)
	print ("get latex", (s7 - s6) * 1000)
