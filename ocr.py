import pytesseract
from pytesseract import Output
import cv2
import sys
import os


def draw_rectangle(image_path, output_path):
	img = cv2.imread(image_path)
	d = pytesseract.image_to_data(img, output_type=Output.DICT)
	n_boxes = len(d['level'])
	for i in range(n_boxes):
		(x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
		cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
	cv2.imwrite(os.path.join("output", output_path), img)


if __name__=='__main__':
	draw_rectangle(sys.argv[1], sys.argv[2])
