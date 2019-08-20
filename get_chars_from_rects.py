from collections import defaultdict 
from extractCharacters import getContours, predict
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

class Rectangle:
	def __init__(self, left, top, right, bottom):
		self.left = left
		self.top = top
		self.right = right
		self.bottom = bottom
		self.paddedPixelMatrix = None
		self.prediction = None
			
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
	return np.zeros((rows + OFFSET * 2, cols + OFFSET * 2), dtype=list)

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
	bounds = defaultdict(lambda: [[], []])
	for row in range(OFFSET, rows + OFFSET):
		for col in range(OFFSET, cols + OFFSET):
			if boxes[row][col] != 0: # boxes[row][col] contains auto_box_id simplified
				points = bounds[boxes[row][col]]
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
	return rectangles
	

#To be used to check if the box creation algo is working fine
def draw_rectangles_on_image(pixels, rectangles):
	for r in rectangles:
		cv2.rectangle(pixels, (r.left, r.top), (r.right, r.bottom), (150, 150, 150))
	cv2.imwrite('modifiedImages/bounds.jpeg', pixels)

#Creates images of the shape the model is trained on. Adds padding if necessary
def create_new_images_from_boxes(pixels, rectangles):
	image_id = 0
	for r in rectangles:
		image_id += 1
		padding = 10
		new_image = np.pad(pixels[r.left : r.right + 1, r.top : r.bottom + 1], (padding, padding), 'constant', constant_values=(255, 255)) 
		try:
			imsave(os.path.join(GEN_FOLDER, '9', str(image_id) + '.jpeg'), new_image)
		except:
			pass
		r.paddedPixelMatrix = new_image
		
if __name__=='__main__':
	from time import time
	start = time()
	model = keras.models.load_model(sys.argv[1])
	pixels = getContours(sys.argv[2])[0]
	rectangles = generate_boxes(pixels)
	create_new_images_from_boxes(pixels, rectangles)
	draw_rectangles_on_image(pixels, rectangles)
	predict(model, rectangles)
	get_latex(rectangles)
	print ("Total time taken", str((time() - start) * 1000) + " ms")
