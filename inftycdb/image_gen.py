import cv2
import sys
import os
import uuid


class ImgDetails:
	def __init__(self, code, x, y, w, h):
		self.code = code
		self.x = int(x)
		self.y = int(y)
		self.w = int(w)
		self.h = int(h)


def gen_images(file_map, src_dir, out_dir):
	for file_name, img_detail_list in file_map.items():
		img = cv2.imread(os.path.join(src_dir, file_name + '.png'))
		for img_detail in img_detail_list:
			cropped_image = img[img_detail.y : img_detail.y + img_detail.h, img_detail.x : img_detail.x + img_detail.w] 
			new_file_name = str(uuid.uuid4()) + '.png'
			cv2.imwrite(os.path.join(out_dir, img_detail.code, new_file_name), cropped_image)


def create_file_map(csv_path):
	file_map = dict()
	with open(csv_path, 'r') as f:
		for l in f.readlines():
			vals = l.split(',')
			code, sheet, cx, cy, width, height = vals[1], vals[2], vals[3], vals[4], vals[5], vals[6]
			try:
				images = file_map[sheet]	
			except:
				images = list()
				file_map[sheet] = images
			images.append(ImgDetails(code, cx, cy, width, height))
	return file_map	


if __name__=='__main__':
	csv_path = sys.argv[1]
	src_dir = sys.argv[2]
	out_dir = sys.argv[3]
	
	file_map = create_file_map(csv_path)
	gen_images(file_map, src_dir, out_dir)
