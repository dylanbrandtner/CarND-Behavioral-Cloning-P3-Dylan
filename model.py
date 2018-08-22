import csv
import cv2
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D, Cropping2D
import sklearn
from sklearn.model_selection import train_test_split
import random

# Read in Driving sim data from given path
def ReadInputSamples(path):
	samples = []
	with open(os.path.join(path,"driving_log.csv")) as csvfile:
		reader = csv.reader(csvfile)
		for line in reader:
			
			#Pull in each of the 3 images into samples connected to adjusted steering data
			correction = 0.2
			samples.append((line[0],float(line[3]))) # Add center image
			samples.append((line[1],float(line[3]) + correction)) # Add left image
			samples.append((line[2],float(line[3]) - correction)) # Add right image
			
	return samples

#Generator for loading samples
def generator(samples, batch_size=32):
	num_samples = len(samples)
	while 1: 
		random.shuffle(samples)
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset:offset+batch_size]
			
			images = []
			angles = []
			for batch_sample in batch_samples:
				images.append(cv2.imread(batch_sample[0]))
				angles.append(batch_sample[1])
			
			# trim image to only see section with road
			X_train = np.array(images)
			y_train = np.array(angles)
			yield sklearn.utils.shuffle(X_train, y_train)
	
# Create a neural network	
def CreateModel():
	model = Sequential()
	
	#Normalization layer
	model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))
	
	#Cropping layer
	model.add(Cropping2D(cropping=((70,25),(0,0))))
	
	#Use network from autonomous driving team at Nvidia
	model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation="relu"))
	model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation="relu"))
	model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation="relu"))
	model.add(Convolution2D(64, 3, 3, activation="relu"))
	model.add(Convolution2D(64, 3, 3, activation="relu"))
	model.add(Flatten())
	model.add(Dense(100))
	model.add(Dense(50))
	model.add(Dense(10))
	model.add(Dense(1))
	
	#Compile network
	model.compile(loss='mse', optimizer='adam')
	return model
	
if __name__ == '__main__':
	
	#Read in and combine all data sets
	samples = []
	for dir in os.listdir("TrainingData"):
		sample_set = ReadInputSamples(os.path.join("TrainingData",dir))		
		samples += sample_set
	
	print("Total samples: " + str(len(samples)))
	
	# Split data and create generators
	train_samples, validation_samples = train_test_split(samples, test_size=0.2)
	train_generator = generator(train_samples, batch_size=32)
	validation_generator = generator(validation_samples, batch_size=32)
	
	#Create and train the model
	model = CreateModel()
	model.fit_generator(train_generator, samples_per_epoch=len(train_samples), 
			validation_data=validation_generator,
			nb_val_samples=len(validation_samples), nb_epoch=20)
	
	#Save the model
	model.save('model.h5')
	