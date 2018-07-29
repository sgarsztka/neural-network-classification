
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K
from imutils import paths
import numpy as np
import argparse
import random
import cv2
import os


def build(width, height, depth, classes):
	model = Sequential()
	inputShape = (height, width, depth)


	if K.image_data_format() == "channels_first":
		inputShape = (depth, height, width)

	model.add(Conv2D(200, (3, 3), padding="same",
		input_shape=inputShape))
	model.add(Activation("relu"))
	model.add(MaxPooling2D(pool_size=(3, 3), strides=(3, 3)))
	model.add(Conv2D(100, (3, 3), padding="same"))
	model.add(Activation("relu"))
	model.add(MaxPooling2D(pool_size=(3, 3), strides=(3, 3)))

	# first (and only) set of FC => RELU layers
	model.add(Flatten())
	model.add(Dense(200))
	model.add(Activation("relu"))

	# softmax classifier
	model.add(Dense(classes))
	model.add(Activation("sigmoid"))


	return model

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to directory with dataset")
ap.add_argument("-m", "--model", required=True,
	help="output nn model file")

args = vars(ap.parse_args())


EPOCHS = 30
INIT_LR = 1e-4
BS = 32
image_count = 0

print("[INFO] loading images...")
data = []
labels = []


imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(imagePaths)


for imagePath in imagePaths:
	image = cv2.imread(imagePath)
	image = cv2.resize(image,(128, 128))
	image = img_to_array(image)
	data.append(image)
	image_count += 1
	print(image_count)
	label = imagePath.split(os.path.sep)[-2]
	label = 1 if label == "ok" else 0
	labels.append(label)


data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)


(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=0.10, random_state=42)

# convert the labels from integers to vectors
trainY = to_categorical(trainY, num_classes=2)
testY = to_categorical(testY, num_classes=2)

aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")


model = build(width=128, height=128, depth=3, classes=2)
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])


H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS, verbose=1)


model.save(args["model"])
