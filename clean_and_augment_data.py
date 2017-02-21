import numpy as np
import cv2
import csv
import time
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.utils import shuffle

from keras.layers.convolutional import Cropping2D, Convolution2D
from keras.layers.core import Activation, Dense, Flatten
from keras.layers.pooling import MaxPooling2D
# import pickle
# import joblib
# from PIL import Image

# File format: ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']

DIRECTORY = '/Users/cvar/selfdrivingcar/term_one/projectthree/data'
DRIVING_FILE = 'driving_log.csv'
DRIVING_FULL_PATH = DIRECTORY + '/' + DRIVING_FILE
FOLDER = 'IMG'
SHRINK_FACTOR = 0.25

# IMG_PATH = DIRECTORY + '/IMG'



def get_img_and_steer_data_from_file():
	steering = []
	center = []
	left = []
	right = []

	# Read in file and save data
	# =============================
	print("=============================")
	print("Reading in file")
	print("=============================")
	start_time = time.time()
	with open(DRIVING_FULL_PATH) as csv_file:
		reader = csv.reader(csv_file)
		next(reader)
		# i = 0
		for row in reader:
			# print (row)
			# print (type(row))
			# if i == 1:
			# 	exit()
			# i += 1

			center.append(row[0])
			left.append(row[1])
			right.append(row[2])
			steering.append(row[3])
	end_time = time.time()

	print("Finished. Elapsed time: {0} secs".format(end_time - start_time))
	print()

	return center, left, right, steering


def augment_steering_data(steering):
	# Manipulate steering data
	# ===========================
	print("=============================")
	print("Creating augmented steering data")
	print("=============================")
	start_time = time.time()
	steering_center = np.array(steering, dtype=np.float32)
	AVG_STEER = np.mean(np.absolute(steering_center))
	CORRECTION = AVG_STEER
	steering_left = steering_center + CORRECTION
	steering_right = steering_center - CORRECTION
	end_time = time.time()
	print("Finished. Elapsed time: {0} secs".format(end_time - start_time))
	print()

	return steering_center, steering_left, steering_right


def test():
	pass
# # Start Write Test
# # ====================
# center_images = []
# print("Processing write test")
# print("================================")
# print()
# start_time = time.time()
# # Load image from file
# img_path = DIRECTORY + '/IMG/center_2016_12_01_13_44_23_207.jpg' 
# img = cv2.imread(img_path)
# # Resize image
# factor = 0.25
# small_img = cv2.resize(img, None, fx=factor, fy=factor, interpolation = cv2.INTER_AREA)


# out_path = DIRECTORY + '/MyImgs/test.jpg'
# cv2.imwrite(out_path, small_img)
# exit()
# end_time = time.time()

# center_images = np.asarray(center_images)
# print("Elapsed time:", end_time - start_time)
# print("Shape & type center images:", center_images.shape, type(center_images))
# print()
# # End Test




def load_center_images(center):
	# Load center images
	# ====================
	center_images = []
	print("=============================")
	print("Processing center images")
	print("=============================")
	start_time = time.time()
	for i in range(len(center)):
		img_path = DIRECTORY + '/' + center[i]
		image = cv2.imread(img_path)
		center_images.append(image)
	end_time = time.time()

	center_images = np.asarray(center_images)
	print("Finished. Elapsed time: {0} secs".format(end_time - start_time))
	print("Shape & type center images:", center_images.shape, type(center_images))
	print()

	return center_images


def load_left_images(left):
	# Load left images
	#===================
	left_images = []
	image_file_suffix_list = left
	print("=============================")
	print("Processing left images")
	print("=============================")
	start_time = time.time()
	for i in range(len(image_file_suffix_list)):
		start = image_file_suffix_list[i].find(FOLDER)
		suffix = image_file_suffix_list[i][start:]
		img_path = DIRECTORY + '/' + suffix
		image = cv2.imread(img_path)
		left_images.append(image)
		if(image is None):
			print("None type for index: {0}, path: {1}".format(i, img_path))
			exit()
	end_time = time.time()

	left_images = np.asarray(left_images)
	print("Finished. Elapsed time: {0} secs".format(end_time - start_time))
	print("Shape & type left images:", left_images.shape, type(left_images))
	print()

	return left_images


def load_right_images(right):
	# Load right images
	# ==================
	right_images = []
	image_file_suffix_list = right
	print("=============================")
	print("Processing right images")
	print("=============================")
	start_time = time.time()
	for i in range(len(image_file_suffix_list)):
		start = image_file_suffix_list[i].find(FOLDER)
		suffix = image_file_suffix_list[i][start:]
		img_path = DIRECTORY + '/' + suffix
		image = cv2.imread(img_path)
		right_images.append(image)
		if(image is None):
			print("None type for index: {0}, path: {1}".format(i, img_path))
			exit()
	end_time = time.time()

	right_images = np.asarray(right_images)
	print("Finished. Elapsed time: {0} secs".format(end_time - start_time))
	print("Shape & type right images:", right_images.shape, type(right_images))
	print()

	return right_images


def stack_data(img_tuple, steer_tuple):
	# Stack data
	# ============
	print("=============================")
	print("Stacking data")
	print("=============================")
	start_time = time.time()
	# images_ctr_lft_rgt = np.concatenate((center_images, left_images, right_images), axis=0)
	# steering_ctr_lft_rgt = np.concatenate((steering_center, steering_left, steering_right), axis=0)
	images_ctr_lft_rgt = np.concatenate(img_tuple, axis=0)
	steering_ctr_lft_rgt = np.concatenate(steer_tuple, axis=0)
	end_time = time.time()

	print("Finished. Elapsed time: {0} secs".format(end_time - start_time))
	print("shape of stacked images:", images_ctr_lft_rgt.shape)
	print("shape of stacked steering:", steering_ctr_lft_rgt.shape)
	print()

	return images_ctr_lft_rgt, steering_ctr_lft_rgt



def create_flipped_images(images, steering):
	# Create augmented data (flipped photos)
	# ======================================
	image_list = []
	steering_list = []
	print("=============================")
	print("Creating augmented data")
	print("=============================")
	start_time = time.time()
	for image, steer in zip(images, steering):
		image2 = np.fliplr(image)
		steer2 = steer * -1.0
		# Append regular
		image_list.append(image)
		steering_list.append(steer)
		# Append flipped
		image_list.append(image2)
		steering_list.append(steer2)
	IMAGES = np.asarray(image_list)
	STEERING = np.asarray(steering_list)
	end_time = time.time()

	print("Finished. Elapsed time: {0} secs".format(end_time - start_time))
	print("shape IMAGES:", IMAGES.shape)
	print("shape STEERING:", STEERING.shape)
	print()

	return IMAGES, STEERING


def shrink_images(images, factor):
	# Shrinking images (photos)
	# ======================================
	print("=============================")
	print("Shrinking images")
	print("=============================")
	start_time = time.time()
	sml_images = []
	for img in images:
		sml_images.append(cv2.resize(img, None, fx=factor, fy=factor, interpolation = cv2.INTER_AREA))
	SML_IMAGES = np.asarray(sml_images)
	end_time = time.time()

	print("Finished. Elapsed time: {0} secs".format(end_time - start_time))
	print("shape of sml_images:", SML_IMAGES.shape)
	print()

	return SML_IMAGES


def shuffle_data(x, y, n_samples=None):
	print("=============================")
	print("Shuffling data")
	print("=============================")
	start_time = time.time()
	x_shuff, y_shuff = shuffle(x, y, n_samples=n_samples)
	end_time = time.time()

	print("Finished. Elapsed time: {0} secs".format(end_time - start_time))
	print("shape of x_shuff:", x_shuff.shape)
	print("shape of y_shuff:", y_shuff.shape)
	print()

	return x_shuff, y_shuff


def write_files():
	print("=============================")
	print("Writting data to files")
	print("=============================")
	# # Save data to file
	# OUT_FILE_PATH = DIRECTORY + '/train.pkl'
	# COMPRESSED_OUT_FILE_PATH = DIRECTORY + '/train_compressed.npz'
	# X_FILE_PATH = DIRECTORY + '/x_train.p'
	# Y_FILE_PATH = DIRECTORY + '/y_train.p'

	# # Write x & y to an npz file
	# start_time = time.time()
	# # np.savez(OUT_FILE_PATH, images=IMAGES, steering=STEERING)
	# # np.savez_compressed(COMPRESSED_OUT_FILE_PATH, images=IMAGES, steering=STEERING)
	# obj = {'images':IMAGES, 'steering': STEERING}
	# joblib.dump(obj, OUT_FILE_PATH)
	# end_time = time.time()
	# print("Wrote data to pkl using joblib in {:.3f} min".format((end_time - start_time)/float(60)))
	# print()


	# # Testing out possibility of writing new images to disk
	# # Not finished yet
	# images_size = len(IMAGES)
	# steering_size = len(STEERING)
	# if(images_size != steering_size):
	# 	print("WARNING: images and steering arrays do not have the same number of rows")

	# IMG_OUT_DIR = DIRECTORY + '/MyImgs'
	# divisor = 10000
	# for i in range(images_size):
	# 	index_string = str(i/float(divisor))[2:]
	# 	file_name = 'drive_img_' + str(i)


def train(images, steering):
	print()
	print("=============================")
	print("=============================")
	print("TRAINING MODEL")
	print("=============================")
	print("=============================")
	start_time = time.time()
	EPOCHS = 2
	BATCH = 500
	TOP_CROP_FACTOR = 0.4
	BOT_CROP_FACTOR = 0.175
	shape = images.shape
	height = shape[1]
	width = shape[2]
	depth = shape[3]
	top_crop = int(TOP_CROP_FACTOR * height)
	bot_crop = int(BOT_CROP_FACTOR * height)
	# Build model
	model = sequential()
	# Layer 1 (5x5): 40x80 -> 18x38
	# Layer 1 (5x5): 64x128 -> 30x62
	model.add(Cropping2D(cropping=((top_crop, bot_crop), (0, 0)), dim_ordering='tf', input_shape=(height, width, depth)))
	model.add(Convolution2D(24, 5, 5, border_mode='valid'))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='same', dim_ordering='tf'))
	# Layer 2 (5x5): 18x38 -> 7x17
	# Layer 2 (5x5): 30x62 -> 13x29
	model.add(Convolution2D(36, 5, 5, border_mode='valid'))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='same', dim_ordering='tf'))
	# Layer 3 (3x3): 7x17 -> 3x8
	# Layer 3 (5x5): 30x62 -> 5x13
	model.add(Convolution2D(48, 3, 3, border_mode='valid'))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='same', dim_ordering='tf'))
	# Layer 4 (2x2): 3x8 -> 2x7 
	# Layer 4 (3x3): 5x13 -> 3x11
	model.add(Convolution2D(64, 2, 2, border_mode='valid'))
	model.add(Activation('relu'))
	# Layer 5 (2x2): 2x7 -> 1x6
	# Layer 5 (3x3): 3x11 -> 1x9
	model.add(Convolution2D(64, 2, 2, border_mode='valid'))
	model.add(Activation('relu'))
	# Flatten: 64x1x6 -> 384
	# Flatten: 64x1x9 -> 576
	model.add(Flatten())
	# Layer 6
	model.add(Dense(375))
    model.add(Activation('relu'))
	# Layer 7
	model.add(Dense(34))
    model.add(Activation('relu'))
	# Layer 8
	model.add(Dense(17))
    model.add(Activation('relu'))
	# Layer 9
	model.add(Dense(10))
    model.add(Activation('relu'))
	# Layer 10
	model.add(Dense(1))
	# Set model training parameters
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    # train model
    history = model.fit(images, steering, nb_epoch=EPOCHS, batch_size=BATCH, validation_split=0.2, shuffle=True)

    print("Finished. Elapsed time: {0} mins".format((end_time - start_time)/float(60)))
	print("shape of x_shuff:", x_shuff.shape)
	print("shape of y_shuff:", y_shuff.shape)
	print()

    print("==============================")
    print("History:")
    print("==============================")
    print(history.history)



def main():
	# Record start time
	module_start = time.time()

	center, left, right, steering = get_img_and_steer_data_from_file()
	steering_center, steering_left, steering_right = augment_steering_data(steering)
	center_images = load_center_images(center)
	left_images = load_left_images(left)
	right_images = load_right_images(right)
	images_ctr_lft_rgt, steering_ctr_lft_rgt = stack_data(
		(center_images, left_images, right_images), 
		(steering_center, steering_left, steering_right))
	sml_images = shrink_images(images_ctr_lft_rgt, SHRINK_FACTOR)
	aug_images, aug_steering  = create_flipped_images(sml_images, steering_ctr_lft_rgt)

	NUM_SAMPLES = 100
	IMAGES, STEERING = shuffle_data(aug_images, aug_steering, NUM_SAMPLES)		
	# IMAGES, STEERING = shuffle_data(aug_images, aug_steering)

	# Record end time
	module_end = time.time()

	print("=============================")
	print("FINISHED")
	print("=============================")
	print("Total time: {:.3f} min".format((module_end - module_start)/float(60)))
	print("FINAL IMAGES SHAPE", IMAGES.shape)
	print("FINAL STEERING SHAPE", STEERING.shape)


if __name__ == '__main__':
	main()
