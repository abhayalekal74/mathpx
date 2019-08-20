from collections import defaultdict 
from extractCharacters import getContours
import json
import uuid
import pickle
from merge_based_on_pixel_pos import get_latex
import math
from scipy.misc import imread, imresize
import cv2
import sys
import numpy as np
from scipy.misc import imsave
import os
import keras


PIX_THRES = 120 
OFFSET = 2 
GEN_FOLDER = 'generated'
MODEL_SHAPE = 50
WHITE_PIXEL = 255
minW = 5
minH = 5


"""
Rectangle class: holds coordinates, prediction and final padded image matrix
"""
class Rectangle:
	def __init__(self, left, top, right, bottom):
		self.left = left
		self.top = top
		self.right = right
		self.bottom = bottom
		self.image_matrix = None
		self.prediction = None


	def get_top_left(self):
		return (self.left, self.top)


	def get_bottom_right(self):
		return (self.right, self.bottom)


#Reading image, apply needed preprocessing steps here
def get_pixels(image_path):
	#img = imread(image_path, 'L')
	#img = cv2.resize(img, None, fx=MAG_FACTOR, fy=MAG_FACTOR, interpolation=cv2.INTER_AREA)
	# Apply dilation and erosion to remove some noise
	#kernel = np.ones((1, 1), np.uint8)
	#img = cv2.dilate(img, kernel, iterations=1)
	#img = cv2.erode(img, kernel, iterations=1)
	#img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
	img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
	img = cv2.GaussianBlur(img, (3,3), 0)
	img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 75, 10)
	cv2.imwrite(os.path.join('modifiedImages', image_path.rsplit('/', 1)[1]), img)	
	return img 


def get_pixel_thres(pixels):
	print (pixels.shape)
	sorted_pix = pixels.flatten()
	sorted_pix.sort()
	for p in sorted_pix:
		print (p)
	a = 0 / 0
	return 120
			


#Print the image highlighting the darker pixels. Update PIXEL_THRES if noise is also drawn
def get_darker_pixel_positions(pixels):
	print ("\nDarker Pixels Representation:")
	avg_px = 0
	for i in range(pixels.shape[0]):
		repr = ""
		for j in range(pixels.shape[1]):
			if pixels[i][j] < PIX_THRES:
				avg_px += pixels[i][j]	
				repr += '.' 
			else:
				repr += ' '
		print (repr)
	print ('avg pixel', avg_px / (pixels.shape[0] * pixels.shape[1]))


def get_adjacent_box_ids(boxes, row, col):
	adjacent_box_ids = []
	for i in range(col - OFFSET, col + OFFSET + 1):
		if boxes[row - 1][i] != 0:
			adjacent_box_ids += boxes[row - 1][i]
	for i in range(col - OFFSET, col):
		if boxes[row][i] != 0:
			adjacent_box_ids += boxes[row][i]
	return list(set(adjacent_box_ids))


#Get empty matrix of the shape of image + 2 * offset as padding
def get_empty_matrix(rows, cols):
	boxes = list()
	for row in range(rows + OFFSET * 2):
		zero_row = list()
		for col in range(cols + OFFSET * 2):
			zero_row.append(0)
		boxes.append(zero_row)
	return boxes


#Assign same box ids to pixels adjacent to each other
def get_adjacent_boxes(rows, cols, pixels, boxes, same_boxes):
	auto_box_id = 1
	for row in range(OFFSET, rows + OFFSET):
		for col in range(OFFSET, cols + OFFSET):
			if (row - OFFSET) < rows and (col - OFFSET) < cols and pixels[row - OFFSET][col - OFFSET] < PIX_THRES:
				adjacent_box_ids = get_adjacent_box_ids(boxes, row, col) 
				if adjacent_box_ids:
					boxes[row][col] = adjacent_box_ids
					for box_id in adjacent_box_ids[1:]:
						same_boxes[box_id] = adjacent_box_ids[0]
				else:
					boxes[row][col] = [auto_box_id]
					auto_box_id += 1


#If boxes with different box ids are adjacent to each other, assign same box ids to both boxes
def simplify_boxes(rows, cols, boxes, same_boxes):
	for row in range(OFFSET, rows + OFFSET):
		for col in range(OFFSET, cols + OFFSET):
			if boxes[row][col] != 0:
				boxes[row][col] = boxes[row][col][0]
				try:
					boxes[row][col] = same_boxes[boxes[row][col]]
				except KeyError:
					pass


#Get rectangle bounds of simplified boxes (left, top, right, bottom)
def get_rect_bounds(rows, cols, boxes):
	bounds = dict()
	for row in range(OFFSET, rows + OFFSET):
		for col in range(OFFSET, cols + OFFSET):
			if boxes[row][col] != 0: # boxes[row][col] contains auto_box_id simplified
				try:
					points = bounds[boxes[row][col]]
				except KeyError:
					points = [[], []]
					bounds[boxes[row][col]] = points
				points[0].append(row - OFFSET)
				points[1].append(col - OFFSET)
	return bounds


#To generate rects by clubbing pixels
def generate_boxes(pixels):
	rows, cols = pixels.shape
	boxes = get_empty_matrix(rows, cols)
	same_boxes = dict()
	get_adjacent_boxes(rows, cols, pixels, boxes, same_boxes)
	simplify_boxes(rows, cols, boxes, same_boxes)
	bounds = get_rect_bounds(rows, cols, boxes)
	rectangles = list()
	for k, v in bounds.items():
		l, t, r, b = min(v[0]), min(v[1]), max(v[0]), max(v[1])
		if (r - l) < minW or (b - t) < minH:
			continue
		rect = Rectangle(l, t, r, b)
		rectangles.append(rect)
	"""
	print ("\nRectangle bounds:")
	for k, v in rectangles.items():
		print ("RectID: {}\tPoints: {}, {}".format(k, v.get_top_left(), v.get_bottom_right()))
	"""
	return rectangles
	

#To be used to check if the box creation algo is working fine
def draw_rectangles_on_image(pixels, rectangles):
	'''
	for rectangle in rectangles:
		cv2.rectangle(pixels, rectangle.get_top_left(), rectangle.get_bottom_right(), (0, 0, 0))
	'''
	for r in rectangles:
		for x in range(r.left, r.right + 1):
			pixels[x][r.top] = 0
			pixels[x][r.bottom] = 0
		for y in range(r.top, r.bottom + 1):
			pixels[r.left][y] = 0
			pixels[r.right][y] = 0


#Creates images of the shape the model is trained on. Adds padding if necessary
def create_new_images_from_boxes(pixels, rectangles):
	folder_name = '9'
	image_id = 0
	for r in rectangles:
		image_id += 1
		padding = 10
		new_image = np.pad(pixels[r.left : r.right + 1, r.top : r.bottom + 1], (padding, padding), 'constant', constant_values=(255, 255)) 
		#magnified_image = cv2.resize(cv2.UMat(new_image), None, fx = MAG_FACTOR, fy = MAG_FACTOR)
		try:
			imsave(os.path.join(GEN_FOLDER, folder_name, str(image_id) + '.jpeg'), new_image)
		except:
			pass
		r.image_matrix = new_image
		
			
# Predict chars in the rectangles detected
def predict(model, rectangles):	
	label_encoder = pickle.load(open('char-recognition-model/label_encoder.pkl', 'rb'))
	with open('char-recognition-model/classcodes.json', 'r') as f:
		class_codes = json.load(f)
	class_argmax = {} # Storing encoding index
	for c in label_encoder.classes_:
		class_argmax[np.argmax(label_encoder.transform([[c,]]))] = c
	for r in rectangles:
		img = imresize(r.image_matrix, (MODEL_SHAPE, MODEL_SHAPE))
		res = model.predict(np.array([img, ]))
		pred = class_codes[class_argmax[np.argmax(res)]]
		r.prediction = pred if len(pred.split(' ')) == 1 else ''
		#print (r.get_top_left(), r.get_bottom_right(), r.prediction)


if __name__=='__main__':
	from time import time
	start = time()
	model = keras.models.load_model(sys.argv[1])
	#pixels = get_pixels(sys.argv[2])
	pixels = getContours(sys.argv[2])[0]
	#PIX_THRES = get_pixel_thres(pixels)
	print ("Pixel Threshold", PIX_THRES)
	#get_darker_pixel_positions(pixels)
	rectangles = generate_boxes(pixels)
	#draw_rectangles_on_image(pixels, rectangles)
	create_new_images_from_boxes(pixels, rectangles)
	predict(model, rectangles)
	get_latex(rectangles)
	#get_darker_pixel_positions(pixels)
	print ("Total time taken", str((time() - start) * 1000) + " ms")
