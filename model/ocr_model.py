from sys import argv, exit
import argparse
import json
from tqdm import tqdm 

from os import listdir
from os.path import isfile, isdir, join

import keras
import numpy as np
from scipy.misc import imread, imresize


def die(err):
	exit("\nError: {}, exiting".format(err))


def validate_args(args):
	# It is mandatory to pass classes json. If not found, the program will exit
	if not args.classes:
		die("classes argument is required")
	try:
		dir_check = isdir(args.classes)
		if not dir_check:
			raise IOError
	except IOError:
		die("classes not found")

	# If predicting and a model is not passed, die
	if args.predict and not args.model:
		die("Pass a model to predict")		
	
	if args.predict and not args.shape:
		die("Please pass the shape on which the model was trained on")

	if args.shape and not args.model:
		print ("Pass a shape only if the model was already trained on this shape, ignoring provided shape")


def parse_args():
	parser = argparse.ArgumentParser()
	required_args = parser.add_argument_group("Required Arguments")
	required_args.add_argument("--classes", help="json containing folder path as the key and class as value")
	parser.add_argument("--model", help="pass an already trained model for further training or for prediction")
	parser.add_argument("--shape", help="shape the images should be resized to, pass if using an already trained model or running prediction. Shape should be the same as what it was trained on. If training a new model, program will print this")
	parser.add_argument("--save-as", dest="saveas", default="output_model.h5", help="save the model as")
	parser.add_argument("--epochs", default=200, type=int)
	parser.add_argument("--no-resize", dest="resize", action="store_false", help="use only if you are absolutely sure that all images are of same shape, program will crash otherwise")
	parser.add_argument("--pred", dest="predict", action="store_true", help="run prediction instead of training")
	parser.set_defaults(predict=False)
	parser.set_defaults(resize=True)
	args = parser.parse_args()
	validate_args(args)
	return args


def get_images(classes):
	images = []
	for folder_name in classes:
		images += [join(folder_name, f) for f in listdir(folder_name) if isfile(join(folder_name, f))]
	return images


def get_classes(src_folder):
	classes = dict()
	for f in listdir(src_folder):
		if isdir(join(src_folder, f)):
			classes[join(src_folder, f)] = f
	print (classes)
	return classes


def get_resize_shape(images, resize, shape=None):
	# If the shape is provided, use as it is else calculate mean rows and cols. 
	if shape:
		rows, cols = map(int, shape.split(","))
	elif resize:	
		row_shapes, col_shapes = [], []
		for img in images:
			img_matrix = imread(img, mode='RGB')
			if img_matrix is not None:
				row_shapes.append(img_matrix.shape[0])
				col_shapes.append(img_matrix.shape[1])
		rows = (int) (np.mean(row_shapes))
		cols = (int) (np.mean(col_shapes))
		print ("Resizing images to shape [{},{}]".format(rows, cols))
	else:
		# Read one image's shape
		for img in images:
			img_matrix = imread(images[0], mode='RGB')
			if img_matrix is not None:
				rows, cols = img_matrix.shape[0], img_matrix.shape[1]
				break
	return rows, cols


class LanguageDetector:

	def __init__(self, args):

		self.classes = get_classes(args.classes)
		self.images = get_images(self.classes)

		self.model_load_from = args.model
		self.save_as = args.saveas
		self.epochs = args.epochs
		self.should_resize_images = args.resize

		self.rows, self.cols = get_resize_shape(self.images, self.should_resize_images, shape=args.shape)

		self.encode_labels()
		self.build_model()


	def encode_labels(self):
		from sklearn.preprocessing import MultiLabelBinarizer 
		labels = []
		# labels in string form are encoded using MultiLabelBinarizer
		for k, v in self.classes.items():
			labels.append([l.strip() for l in v.split(",")])
		self.label_encoder = MultiLabelBinarizer()
		print ("Transforms: {}".format(self.label_encoder.fit_transform(labels)))		
		print ("0utput classes: {}".format(self.label_encoder.classes_))
 

	def get_dataset(self, start, end):
		x, y = None,[]
		for i in tqdm(range(start, end)):
			img = self.images[i]
			# Read the image as a matrix in RGB mode
			img_matrix = imread(img, mode='RGB')
			if img_matrix is not None:
				if self.should_resize_images:
					# Resize the images to a common shape
					img_matrix = imresize(img_matrix, (self.rows, self.cols, 3)) 
				# Append the image matrix to the list of input matrices
				if args.predict:
					if x is None:
						x = []
					x.append(img_matrix)
				else:
					if x is None:
						x = np.array([img_matrix])
					else:
						x = np.append(x, [img_matrix], axis = 0)
				#x.append(img_matrix)
				labels = str(self.classes[img[:img.rfind("/")]]).split(",")
				# Append the labels to output list
				y.append(np.squeeze(self.label_encoder.transform([labels])))

		if args.predict:
			x = np.array(x)
		print ("Get dataset shape", x.shape)
	   
		#x -= 214.8779066439155 
		#x /= 15.826097547930841 
		return x, np.array(y)


	def build_model(self):
		if self.model_load_from:
			# If a model has been passed, use it.
			self.model = keras.models.load_model(self.model_load_from)
		else:
			# Otherwise build a CNN
			self.model = keras.models.Sequential()
	
			self.model.add(keras.layers.BatchNormalization(input_shape=(self.rows, self.cols, 3)))
	
			self.model.add(keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(self.rows, self.cols, 3))) 
			self.model.add(keras.layers.BatchNormalization())
			self.model.add(keras.layers.MaxPooling2D(pool_size=(2,2), padding='same'))
			self.model.add(keras.layers.Dropout(0.2))
	
			self.model.add(keras.layers.Conv2D(24, (3,3), activation='relu', input_shape=(self.rows, self.cols, 3))) 
			self.model.add(keras.layers.BatchNormalization())
			self.model.add(keras.layers.MaxPooling2D(pool_size=(2,2), padding='same'))
			self.model.add(keras.layers.Dropout(0.2))
	
			self.model.add(keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(self.rows, self.cols, 3))) 
			self.model.add(keras.layers.BatchNormalization())
			self.model.add(keras.layers.MaxPooling2D(pool_size=(2,2), padding='same'))
			self.model.add(keras.layers.Dropout(0.2))
	
			self.model.add(keras.layers.Conv2D(40, (3,3), activation='relu', input_shape=(self.rows, self.cols, 3))) 
			self.model.add(keras.layers.BatchNormalization())
			self.model.add(keras.layers.MaxPooling2D(pool_size=(2,2), padding='same'))
			self.model.add(keras.layers.Dropout(0.2))

			# Flattening the input to be passed onto Fully Connected Layers
			self.model.add(keras.layers.Flatten())

			# First fully connected layer
			self.model.add(keras.layers.Dense(1024, activation='relu'))
			self.model.add(keras.layers.Dropout(0.2))
			self.model.add(keras.layers.Dense(840, activation='relu'))
			self.model.add(keras.layers.Dropout(0.2))

			# Last layer, responsible for predicting the output
			self.model.add(keras.layers.Dense(len(self.label_encoder.classes_), activation='softmax'))

	def train(self):
		self.model.compile(loss='categorical_crossentropy',
					  optimizer='adam',
					  metrics=['accuracy'])

		# Create checkpoint after every epoch
		cb = [keras.callbacks.ModelCheckpoint(self.save_as[:-3] + "_cp.h5", monitor='acc', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)]

		x_train, y_train = self.get_dataset(5, len(self.images))
		x_val, y_val = self.get_dataset(0, 5)

		# Fit the model on the training data
		self.model.fit(x_train, y_train,
				  epochs=self.epochs,
				  batch_size=32,
				  callbacks=cb
				)

		# Save the model with the value passed for -saveas argument
		self.model.save(self.save_as)

		print ("\nEvaluation on validation data: {}".format(dict(zip(["Loss", "Accuracy"], self.model.evaluate(x_val, y_val, batch_size=32)))))


	def predict(self):
		x_test, y_test = self.get_dataset(0, len(self.images))
		res = self.model.predict(x_test, batch_size=32, verbose=1)
		class_argmax = {} # Storing encoding index
		for c in self.label_encoder.classes_:
			class_argmax[np.argmax(self.label_encoder.transform([[c,]]))] = c
		for i in range(len(self.images)):
			print (self.images[i], np.argmax(res[i]))
		print ("\nEvaluation on test data: {}".format(dict(zip(["Loss", "Accuracy"], self.model.evaluate(x_test, y_test, batch_size=32)))))


if __name__=="__main__":
	args = parse_args()
	keras.backend.set_learning_phase(0)
	language_detector = LanguageDetector(args)
	language_detector.predict() if args.predict else language_detector.train()
