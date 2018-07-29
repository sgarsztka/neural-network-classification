
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from skimage.util.shape import view_as_windows
import numpy as np
import argparse
import imutils
import cv2
import os


ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to model file")
ap.add_argument("-d", "--image", required=False,
	help="path to directory with images for test")
args = vars(ap.parse_args())





def load_images_from_folder(folder):
	images = []
	for filename in os.listdir(folder):
		img = cv2.imread(os.path.join(folder,filename))
		if img is not None:
			images.append(img)

	return images

folder = (args ["image"])
model = load_model(args["model"])
imgs = load_images_from_folder(folder)
k=(len(imgs))
image_flag = 0
buffer = []
buffer_full = []
print(k)

for image in imgs:
	buffer_full.append(image)
	orig = image.copy()


for patch_image in buffer_full:
	#image = cv2.imread(args["image"])
	patch_image = cv2.resize(patch_image, (128, 128))
	print(len(buffer_full))
	orig = patch_image.copy()
	patch_image = patch_image.astype("float") / 255.0
	patch_image = img_to_array(patch_image)
	patch_image = np.expand_dims(patch_image, axis=0)
	(nok, ok) = model.predict(patch_image, verbose=1)[0]

	pos= "{:2f}".format(ok*100)
	neg="{:2f}".format(nok*100)

	output = imutils.resize(orig, width=250)
	cv2.putText(output, pos, (15, 25),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
	cv2.putText(output, neg, (30, 45),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
	cv2.imshow("wynik", output)
	cv2.waitKey(0)
