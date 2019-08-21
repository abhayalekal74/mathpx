from sys import argv, exit
from pprint import PrettyPrinter
import argparse
import json
from tqdm import tqdm 
import pickle

from os import listdir
from os.path import isfile, isdir, join

import keras
import numpy as np
from scipy.misc import imread, imresize


predict = False
pp = PrettyPrinter(indent=4)

def die(err):
	exit("\nError: {}, exiting".format(err))


def validateArgs(args):
	# It is mandatory to pass classes json. If not found, the program will exit
	if not args.classes:
		die("classes argument is required")
	try:
		dirCheck = isdir(args.classes)
		if not dirCheck:
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


def parseArgs():
	parser = argparse.ArgumentParser()
	requiredArgs = parser.add_argument_group("Required Arguments")
	requiredArgs.add_argument("--classes", help="json containing folder path as the key and class as value")
	parser.add_argument("--model", help="pass an already trained model for further training or for prediction")
	parser.add_argument("--shape", help="shape the images should be resized to, pass if using an already trained model or running prediction. Shape should be the same as what it was trained on. If training a new model, program will print this")
	parser.add_argument("--save-as", dest="saveas", default="output_model.h5", help="save the model as")
	parser.add_argument("--epochs", default=200, type=int)
	parser.add_argument("--no-resize", dest="resize", action="store_false", help="use only if you are absolutely sure that all images are of same shape, program will crash otherwise")
	parser.add_argument("--pred", dest="predict", action="store_true", help="run prediction instead of training")
	parser.set_defaults(predict=False)
	parser.set_defaults(resize=True)
	args = parser.parseArgs()
	validateArgs(args)
	return args


def getImages(classes):
	images = []
	for folderName in classes:
		images += [join(folderName, f) for f in listdir(folderName) if isfile(join(folderName, f))]
	return images


def getClasses(srcFolder):
	classes = dict()
	for f in listdir(srcFolder):
		if isdir(join(srcFolder, f)):
			classes[join(srcFolder, f)] = f
	return classes


def getResizeShape(images, resize, shape=None):
	# If the shape is provided, use as it is else calculate mean rows and cols. 
	if shape:
		rows, cols = map(int, shape.split(","))
	elif resize:	
		rowShapes, colShapes = [], []
		for img in images:
			imgMatrix = imread(img, mode='L')
			if imgMatrix is not None:
				rowShapes.append(imgMatrix.shape[0])
				colShapes.append(imgMatrix.shape[1])
		rows = (int) (np.mean(rowShapes))
		cols = (int) (np.mean(colShapes))
		print ("Resizing images to shape [{},{}]".format(rows, cols))
	else:
		# Read one image's shape
		for img in images:
			imgMatrix = imread(images[0], mode='L')
			if imgMatrix is not None:
				rows, cols = imgMatrix.shape[0], imgMatrix.shape[1]
				break
	return rows, cols


class LanguageDetector:

	def __init__(self, args):

		self.classes = getClasses(args.classes)
		self.images = getImages(self.classes)

		self.modelLoadFrom = args.model
		self.saveas = args.saveas
		self.epochs = args.epochs
		self.shouldResizeImages = args.resize

		self.rows, self.cols = getResizeShape(self.images, self.shouldResizeImages, shape=args.shape)

		self.encodeLabels()
		self.buildModel()


	def encodeLabels(self):
		from sklearn.preprocessing import MultiLabelBinarizer 
		if predict:
			self.labelEncoder = pickle.load(open('labelEncoder.pkl', 'rb'))
		else:
			labels = []
			# labels in string form are encoded using MultiLabelBinarizer
			for k, v in self.classes.items():
				labels.append([l.strip() for l in v.split(",")])
			self.labelEncoder = MultiLabelBinarizer()
			self.labelEncoder.fit_transform(labels)		
			pickle.dump(self.labelEncoder, open('labelEncoder.pkl', 'wb'))
 

	def getDataset(self, start, end):
		x, y = None,[]
		temp = []
		for i in tqdm(range(start, end)):
			img = self.images[i]
			imgMatrix = imread(img, mode='L')
			if imgMatrix is not None:
				if self.shouldResizeImages:
					# Resize the images to a common shape
					imgMatrix = imresize(imgMatrix, (self.rows, self.cols)) 
				# Append the image matrix to the list of input matrices
				if args.predict:
					if x is None:
						x = []
					x.append(imgMatrix)
				else:
					temp.append(imgMatrix)
					if len(temp) % 10000 == 0:
						if x is None:
							x = np.array(temp)
						else:
							x = np.append(x, np.array(temp), axis = 0)
						temp = []
				#x.append(imgMatrix)
				labels = str(self.classes[img[:img.rfind("/")]]).split(",")
				# Append the labels to output list
				y.append(np.squeeze(self.labelEncoder.transform([labels])))
		print (x is None)
		if len(temp) > 0:
			if x is None:
				x = np.array(temp)
			else:
				x = np.append(x, np.array(temp), axis = 0)

		if args.predict:
			x = np.array(x)
		print ("Get dataset shape", x.shape)
	   
		#x -= 214.8779066439155 
		#x /= 15.826097547930841 
		return x, np.array(y)


	def buildModel(self):
		if self.modelLoadFrom:
			# If a model has been passed, use it.
			self.model = keras.models.load_model(self.modelLoadFrom)
		else:
			# Otherwise build a CNN
			self.model = keras.models.Sequential()
	
			self.model.add(keras.layers.BatchNormalization(input_shape=(self.rows, self.cols)))
	
			self.model.add(keras.layers.Conv1D(16, 5, activation='relu', input_shape=(self.rows, self.cols))) 
			self.model.add(keras.layers.BatchNormalization())
			self.model.add(keras.layers.MaxPooling1D(pool_size=2, padding='same'))
			self.model.add(keras.layers.Dropout(0.2))
	
			self.model.add(keras.layers.Conv1D(32, 5, activation='relu'))
			self.model.add(keras.layers.BatchNormalization())
			self.model.add(keras.layers.MaxPooling1D(pool_size=2, padding='same'))
			self.model.add(keras.layers.Dropout(0.2))
	
			self.model.add(keras.layers.Conv1D(48, 3, activation='relu'))
			self.model.add(keras.layers.BatchNormalization())
			self.model.add(keras.layers.MaxPooling1D(pool_size=2, padding='same'))
			self.model.add(keras.layers.Dropout(0.2))
	
			self.model.add(keras.layers.Conv1D(64, 3, activation='relu'))
			self.model.add(keras.layers.BatchNormalization())
			self.model.add(keras.layers.MaxPooling1D(pool_size=2, padding='same'))
			self.model.add(keras.layers.Dropout(0.2))
	
			# Flattening the input to be passed onto Fully Connected Layers
			self.model.add(keras.layers.Flatten())

			# First fully connected layer
			self.model.add(keras.layers.Dense(1024, activation='relu'))
			self.model.add(keras.layers.Dropout(0.2))
			self.model.add(keras.layers.Dense(840, activation='relu'))
			self.model.add(keras.layers.Dropout(0.2))

			# Last layer, responsible for predicting the output
			self.model.add(keras.layers.Dense(len(self.labelEncoder.classes_), activation='softmax'))

	def train(self):
		self.model.compile(loss='categorical_crossentropy',
					  optimizer='rmsprop',
					  metrics=['accuracy'])

		# Create checkpoint after every epoch
		cb = [keras.callbacks.ModelCheckpoint(self.saveas[:-3] + "_cp.h5", monitor='acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)]

		XTrain, YTrain = self.getDataset(5, len(self.images))
		XVal, YVal = self.getDataset(0, 5)

		# Fit the model on the training data
		self.model.fit(XTrain, YTrain,
				  epochs=self.epochs,
				  batch_size=32,
				  callbacks=cb
				)

		# Save the model with the value passed for -saveas argument
		self.model.save(self.saveas)

		print ("\nEvaluation on validation data: {}".format(dict(zip(["Loss", "Accuracy"], self.model.evaluate(XVal, YVal, batch_size=32)))))


	def predict(self):
		with open('classcodes.json', 'r') as f:
			class_codes = json.load(f)
		XTest, YTest = self.getDataset(0, len(self.images))
		res = self.model.predict(XTest, batch_size=32, verbose=1)
		classArgmax = {} # Storing encoding index
		print (self.labelEncoder.classes_)
		for c in self.labelEncoder.classes_:
			classArgmax[np.argmax(self.labelEncoder.transform([[c,]]))] = c
		outputs = list()
		for i in range(len(self.images)):
			try:
				outputs.append((int(self.images[i].rsplit('/', 1)[1].split('.')[0]), self.images[i], classArgmax[np.argmax(res[i])], class_codes[classArgmax[np.argmax(res[i])]]))
			except:
				outputs.append((self.images[i].rsplit('/', 1)[1].split('.')[0], self.images[i], classArgmax[np.argmax(res[i])], class_codes[classArgmax[np.argmax(res[i])]]))
				
		outputs.sort(key=lambda x: x[0])
		pp.pprint (outputs)
		print ("\nEvaluation on test data: {}".format(dict(zip(["Loss", "Accuracy"], self.model.evaluate(XTest, YTest, batch_size=32)))))


if __name__=="__main__":
	args = parseArgs()
	predict = args.predict
	print (predict)
	keras.backend.set_learning_phase(0)
	languageDetector = LanguageDetector(args)
	languageDetector.predict() if args.predict else languageDetector.train()
