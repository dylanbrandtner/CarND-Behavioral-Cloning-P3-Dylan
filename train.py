import csv
import cv2
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense

# Read in Driving sim data from given path
def ReadInputData(path):
	lines = []
	images = []
	measurements = []
	with open(os.path.join(path,"driving_log.csv")) as csvfile:
		reader = csv.reader(csvfile)
		for line in reader:
			lines.append(line)
	
	for line in lines:
		filename = line[0].split('/')[-1]
		path = os.path.join("..", "data", "IMG", filename)
		images.append(cv2.imread(path))
		measurements.append(float(line[3]))
		
	return images,measurements

# Create a neural network	
def CreateModel():
	model = Sequential()
	model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))
	model.add(Flatten())
	model.add(Dense(1))
	
	model.compile(loss='mse', optimizer='adam')
	return model
	
if __name__ == '__main__':
	
	#Read in and combine all data sets
	X_train_in = []
	Y_train_in = []
	for dir in os.listdir("TrainingData"):
		images,measurements = ReadInputData(os.path.join("TrainingData",dir))		
		X_train_in += images
		Y_train_in += measurements
	
	X_train = np.array(X_train_in)
	Y_train = np.array(Y_train_in)
	
	#Create and train the model
	model = CreateModel()
	model.fit(X_train, Y_train, validation_split=0.2, shuffle=True, nb_epoch=5)
	
	#Save the model
	model.save('model.h5')
	