import keras
import sys


if __name__=='__main__':
	model = keras.models.load_model(sys.argv[1])
	print (model.summary())
