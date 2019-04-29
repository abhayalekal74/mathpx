import cv2


vertical_thres = 6 
horiz_thres = 6
pixel_thres = 145 


class Rectangle:
	def __init__(self, left, top, right, bottom):
		self.left = left
		self.top = top
		self.right = right
		self.bottom = bottom


	def get_top_left(self):
		return (self.left, self.top)


	def get_bottom_right(self):
		return (self.right, self.bottom)



def get_pixels(image_path):
	return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)


def get_darker_pixel_positions(pixels):
	print ("\nDarker Pixels Representation:")
	for i in range(pixels.shape[0]):
		repr = ""
		for j in range(pixels.shape[1]):
			if pixels[i][j] < pixel_thres:
				repr += '*' 
			else:
				repr += '.'
		print (repr)


def get_adjacent_box_ids(boxes, row, col):
	adjacent_box_ids = []
	if boxes[row - 1][col - 1] != 0:
		adjacent_box_ids += boxes[row - 1][col - 1]
	if boxes[row - 1][col] != 0:
		adjacent_box_ids += boxes[row - 1][col]
	if boxes[row - 1][col + 1] != 0:
		adjacent_box_ids += boxes[row - 1][col + 1]
	if boxes[row][col - 1] != 0:
		adjacent_box_ids += boxes[row][col - 1]
	return list(set(adjacent_box_ids))


def get_empty_matrix(rows, cols):
	boxes = list()
	for row in range(rows + 2):
		zero_row = list()
		for col in range(cols + 2):
			zero_row.append(0)
		boxes.append(zero_row)
	return boxes


def get_adjacent_boxes(rows, cols, pixels, boxes, same_boxes):
	auto_box_id = 1
	for row in range(1, rows + 1):
		for col in range(1, cols + 1):
			if (row - 1) < rows and (col - 1) < cols and pixels[row - 1][col - 1] < pixel_thres:
				adjacent_box_ids = get_adjacent_box_ids(boxes, row, col) 
				if adjacent_box_ids:
					boxes[row][col] = adjacent_box_ids
					for box_id in adjacent_box_ids[1:]:
						same_boxes[box_id] = adjacent_box_ids[0]
				else:
					boxes[row][col] = [auto_box_id]
					auto_box_id += 1


def simplify_boxes(rows, cols, boxes, same_boxes):
	for row in range(1, rows + 1):
		for col in range(1, cols + 1):
			if boxes[row][col] != 0:
				boxes[row][col] = boxes[row][col][0]
				try:
					boxes[row][col] = same_boxes[boxes[row][col]]
				except KeyError:
					pass


def get_rect_bounds(rows, cols, boxes):
	bounds = dict()
	for row in range(1, rows + 1):
		for col in range(1, cols + 1):
			if boxes[row][col] != 0:
				try:
					points = bounds[boxes[row][col]]
				except KeyError:
					points = [[], []]
					bounds[boxes[row][col]] = points
				points[0].append(row - 1)
				points[1].append(col - 1)
	return bounds


def generate_boxes(pixels):
	rows, cols = pixels.shape
	boxes = get_empty_matrix(rows, cols)
	same_boxes = dict()
	get_adjacent_boxes(rows, cols, pixels, boxes, same_boxes)
	simplify_boxes(rows, cols, boxes, same_boxes)
	bounds = get_rect_bounds(rows, cols, boxes)
	rectangles = dict()
	for k, v in bounds.items():
		rect = Rectangle(min(v[0]), min(v[1]), max(v[0]), max(v[1]))	
		rectangles[k] = rect	
	print ("\nRectangle bounds:")
	for k, v in rectangles.items():
		print ("RectID: {}\tPoints: {}, {}".format(k, v.get_top_left(), v.get_bottom_right()))
	return rectangles
	

def draw_rectangles_on_image(pixels, rectangles):
	'''
	for rectangle in rectangles.values():
		cv2.rectangle(pixels, rectangle.get_top_left(), rectangle.get_bottom_right(), (0, 0, 0))
	'''
	for r in rectangles.values():
		for x in range(r.left, r.right + 1):
			pixels[x][r.top] = 0
			pixels[x][r.bottom] = 0
		for y in range(r.top, r.bottom + 1):
			pixels[r.left][y] = 0
			pixels[r.right][y] = 0


if __name__=='__main__':
	import sys
	pixels = get_pixels(sys.argv[1])
	get_darker_pixel_positions(pixels)
	rectangles = generate_boxes(pixels)
	draw_rectangles_on_image(pixels, rectangles)
	get_darker_pixel_positions(pixels)

