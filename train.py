import numpy as np
import cv2
import csv
import time
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from scipy import stats

from keras.models import Sequential
from keras.layers.convolutional import Cropping2D, Convolution2D
from keras.layers.core import Activation, Dense, Flatten, Dropout
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import SGD, Adam
# import pickle
# import joblib
# from PIL import Image

# File format: ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']

DIRECTORY = '/Users/cvar/selfdrivingcar/term_one/projectthree/data'
DRIVING_FILE = 'driving_log.csv'
DRIVING_FULL_PATH = DIRECTORY + '/' + DRIVING_FILE
FOLDER = 'IMG'
SHRINK_FACTOR = 0.4


# Model params
# ==================
# ==================
EPOCHS = 10
BATCH = 210
TOP_CROP_FACTOR = 60.0/160.0
BOT_CROP_FACTOR = 25.0/160.0
CONV_FILTER = (5, 5, 5, 3, 3) # Nvidia
# CONV_FILTER = (3, 3, 3, 2, 2)

# CONV_DEPTH = (24, 36, 48, 64, 64) # Nvidia
# CONV_DEPTH = (2, 4, 6, 8, 10)
# CONV_DEPTH = (4, 6, 8, 10, 12)
# CONV_DEPTH = (8, 12, 16, 20, 24)
CONV_DEPTH = (2, 4, 8, 16, 32)
# CONV_DEPTH = (16, 24, 32, 40, 48)
# CONV_DEPTH = (12, 24, 36, 48, 64)

FIRST_DENSE_PARAM = 400 # 100 -> Nvidia
# DENSE_PARAMS = (200, 100, 10, 1)
DENSE_PARAMS = (400, 100, 25, 1)
LR = 0.001
DECAY = 0.1
DROP_PROB = 0.5


# Params for testing
# ==================
# ==================
TEST_SIZE = 210 * 1 
NUM_SAMPLES = 100 # Required if using the shuffle statement that inclucde the third parameter n_samples

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


def indices_of_desired_data(steering, p_zero_keep):
	"""
	Only keep p_zero_keep proportion of zero angle steering data
	"""
	print("=============================")
	print("Finding indicies to keep")
	print("=============================")
	start_time = time.time()

	reg_indices = []
	zero_indices = []

	for i, steer in enumerate(steering):
		if abs(steer) < 0.001:
			zero_indices.append(i)
		else:
			reg_indices.append(i)

	size_reg = len(reg_indices)
	size_zero = len(zero_indices)
	size_total = size_reg + size_zero
	print("Reg: {0} -> {1:.2f}".format(size_reg, size_reg/size_total))
	print("Zero: {0} -> {1:.2f}".format(size_zero, size_zero/size_total))
	print()

	n_zeros = int(p_zero_keep * len(zero_indices))
	zero_keep_indices = shuffle(zero_indices, n_samples=n_zeros)

	keep_indices = np.concatenate((reg_indices, zero_keep_indices))
	print("Total Keep:", len(keep_indices))
	end_time = time.time()

	print("Finished. Elapsed time: {0} secs".format(end_time - start_time))

	return keep_indices


def filter_data(lists, indicies):
	# Filtering lists to only include the indices supplied in the parameter "indices"
	# =============================
	print("=============================")
	print("Filtering lists")
	print("=============================")
	start_time = time.time()

	out_lists = []
	num_lists = len(lists)
	for i in range (num_lists):
		out_lists.append([])

	for index in indicies:
		for j in range(num_lists):
			out_lists[j].append(lists[j][index])

	end_time = time.time()

	print("Finished. Elapsed time: {0} secs".format(end_time - start_time))

	return out_lists



def augment_steering_data(steering):
	# Manipulate steering data
	# ===========================
	print("=============================")
	print("Creating augmented steering data")
	print("=============================")
	start_time = time.time()
	steering_center = np.array(steering, dtype=np.float32)
	AVG_STEER = np.mean(np.absolute(steering_center))
	print("MIN STEER:", steering_center.min())
	print("MAX STEER:", steering_center.max())
	# print("MODE STEER:", stats.mode(steering_center))
	CORRECTION = AVG_STEER * 1.0
	print("AVG_STEER:", AVG_STEER)
	print("CORRECTION:", CORRECTION)
	steering_left = steering_center.copy() + CORRECTION
	steering_right = steering_center.copy() - CORRECTION
	# print("AVG steering_left", np.mean(steering_left))
	# print("MIN steering_left", steering_left.min())
	# print("MAX steering_left", steering_left.max())
	# print("AVG steering_right", np.mean(steering_right))
	# print("MIN steering_right", steering_right.min())
	# print("MAX steering_right", steering_right.max())
	end_time = time.time()
	print("Finished. Elapsed time: {0} secs".format(end_time - start_time))
	print()

	return steering_center, steering_left, steering_right


def test():
	# Start Write Test
	# ====================
	center_images = []
	print("Processing write test")
	print("================================")
	print()
	start_time = time.time()
	# Load image from file
	img_path = DIRECTORY + '/IMG/center_2016_12_01_13_44_23_207.jpg' 
	img = cv2.imread(img_path)
	cv2.imshow('image', img)
	input()
	exit()
	# # Resize image
	# factor = 0.25
	# small_img = cv2.resize(img, None, fx=factor, fy=factor, interpolation = cv2.INTER_AREA)

	img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY);
	image_data = img_gray.copy()
	print(image_data[34, 78])
	a = -0.5
	b = 0.5
	grayscale_min = 0
	grayscale_max = 255
	img_norm = a + ( ( (image_data - grayscale_min)*(b - a) )/( grayscale_max - grayscale_min ) )
	# cv2.normalize(img_gray, img_gray)
	print(img_norm[34, 78])

	out_path = DIRECTORY + '/MyImgs/test.jpg'
	cv2.imwrite(out_path, img_gray)
	end_time = time.time()

	# center_images = np.asarray(center_images)
	print("Test Finished. Elapsed time:", end_time - start_time)
	# print("Shape & type center images:", center_images.shape, type(center_images))
	print()
	exit()
	# End Test




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


def drive_preprocess(image):
	image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
	image = cv2.resize(image, None, fx=SHRINK_FACTOR, fy=SHRINK_FACTOR, interpolation = cv2.INTER_AREA)
	image = to_zero_one(image)
	shape = image.shape
	image = image.reshape(shape[0], shape[1], 1)

	return image


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


def to_grayscale(images):
	# Shrinking images (photos)
	# ======================================
	print("=============================")
	print("Converting to grayscale")
	print("=============================")
	start_time = time.time()
	gray_images = []
	for img in images:
		gray_images.append(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
	GRAY_IMAGES = np.asarray(gray_images)
	end_time = time.time()

	print("Finished. Elapsed time: {0} secs".format(end_time - start_time))
	print("shape of gray_images:", GRAY_IMAGES.shape)
	print()

	return GRAY_IMAGES


def normalize(images):
	# Shrinking images (photos)
	# ======================================
	print("=============================")
	print("Normalizing images")
	print("=============================")
	start_time = time.time()
	norm_images = []
	for img in images:
		norm_images.append(to_zero_one(img))
	NORM_IMAGES = np.asarray(norm_images)
	end_time = time.time()

	print("Finished. Elapsed time: {0} secs".format(end_time - start_time))
	print("shape of NORM_IMAGES:", NORM_IMAGES.shape)
	print()

	return NORM_IMAGES


def to_zero_one(image):
	a = -0.5
	b = 0.5
	grayscale_min = 0
	grayscale_max = 255
	return a + ( ( (image - grayscale_min)*(b - a) )/( grayscale_max - grayscale_min ) )


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


def look_at_data(steering):
	q = [10, 20, 30, 40, 50, 60, 70, 80, 90] 
	print("Steering percentiles:")
	percentiles = np.percentile(steering, q)
	for i, percentile in enumerate(percentiles):
		print(((i+1)*10, percentile))
	print("Steering mean:", np.mean(steering))
	print("Steering min:", steering.min())
	print("Steering max:", steering.max())
	print("Steering mode:", stats.mode(steering))
	print("Steering size:", len(steering))
	print("Steering std:", np.std(steering))
	print()
	exit()


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


def train_model(images, steering):
	print()
	print("=============================")
	print("=============================")
	print("TRAINING MODEL ()")
	print("=============================")
	print("=============================")
	start_time = time.time()
	shape = images.shape
	height = shape[1]
	width = shape[2]
	depth = shape[3]
	top_crop = int(TOP_CROP_FACTOR * height)
	bot_crop = int(BOT_CROP_FACTOR * height)


	n_samples = shape[0]
	n_test = int(0.17 * n_samples)
	test_x = images[:n_test]
	test_y = steering[:n_test]
	train_x = images[n_test:]
	train_y = steering[n_test:]


	# Build model
	model = Sequential()


	# Layer 1 (5x5): 40x80 -> 18x38
	# Layer 1 (5x5): 64x128 -> 30x62
	# model.add(Cropping2D(cropping=((top_crop, bot_crop), (0, 0)), dim_ordering='tf', input_shape=(height, width, depth)))
	# model.add(Convolution2D(24, 5, 5, border_mode='valid'))
	model.add(Convolution2D(CONV_DEPTH[0], CONV_FILTER[0], CONV_FILTER[0], dim_ordering='tf', border_mode='valid', input_shape=(height, width, depth)))
	# model.add(Convolution2D(24, CONV_FILTER[0], CONV_FILTER[0], border_mode='valid', dim_ordering='tf', input_shape=(height, width, 1)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='same', dim_ordering='tf'))


	# Dropout
	# model.add(Dropout(DROP_PROB))


	# Layer 2 (5x5): 18x38 -> 7x17
	# Layer 2 (5x5): 30x62 -> 13x29
	model.add(Convolution2D(CONV_DEPTH[1], CONV_FILTER[1], CONV_FILTER[1], border_mode='valid', dim_ordering='tf'))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='same', dim_ordering='tf'))


	# Dropout
	# model.add(Dropout(DROP_PROB))


	# Layer 3 (3x3): 7x17 -> 3x8
	# Layer 3 (5x5): 30x62 -> 5x13
	model.add(Convolution2D(CONV_DEPTH[2], CONV_FILTER[2], CONV_FILTER[2], border_mode='valid', dim_ordering='tf'))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='same', dim_ordering='tf'))


	# Dropout
	# model.add(Dropout(DROP_PROB))


	# Layer 4 (2x2): 3x8 -> 2x7 
	# Layer 4 (3x3): 5x13 -> 3x11
	model.add(Convolution2D(CONV_DEPTH[3], CONV_FILTER[3], CONV_FILTER[3], border_mode='valid', dim_ordering='tf'))
	model.add(Activation('relu'))


	# Dropout
	# model.add(Dropout(DROP_PROB))


	# Layer 5 (2x2): 2x7 -> 1x6
	# Layer 5 (3x3): 3x11 -> 1x9
	model.add(Convolution2D(CONV_DEPTH[4], CONV_FILTER[4], CONV_FILTER[4], border_mode='valid', dim_ordering='tf'))
	model.add(Activation('relu'))


	# Dropout
	# model.add(Dropout(DROP_PROB))



	# Flatten: 64x1x6 -> 384
	# Flatten: 64x1x9 -> 576
	model.add(Flatten())


	# Layer 6
	model.add(Dense(FIRST_DENSE_PARAM))
	model.add(Activation('relu'))


	# Dropout
	# model.add(Dropout(DROP_PROB))


	# Layer 7
	model.add(Dense(int(FIRST_DENSE_PARAM*0.5)))
	model.add(Activation('relu'))


	# Dropout
	# model.add(Dropout(DROP_PROB))


	# Layer 8
	model.add(Dense(10))
	model.add(Activation('relu'))


	# Layer 9
	model.add(Dense(1))


	# Set model training parameters
	# sgd = SGD(lr=LR, decay=DECAY)
	# model.compile(optimizer=sgd, loss='mse', metrics=['accuracy'])
	adam = Adam(lr=LR)
	# model.compile(optimizer=adam, loss='mean_squared_error', metrics=['accuracy', 'mean_absolute_error'])
	model.compile(optimizer=adam, loss='mean_absolute_error')
    # Train model
	history = model.fit(train_x, train_y, nb_epoch=EPOCHS, batch_size=BATCH, validation_split=0.2, shuffle=True)
	# Record end time
	end_time = time.time()

	print("Finished. Elapsed time: {0} mins".format((end_time - start_time)/float(60)))
	print()

	print("==============================")
	print("History:")
	print("==============================")
	for key, val in history.history.items():
		print("{0}: {1:.3f}".format(key, val[EPOCHS-1]))
	print()
	# print(history.history)

	print("==============================")
	print("Predict:")
	print("==============================")
	# test_x, test_y = shuffle(images, steering, n_samples=(96))
	test_pred = model.predict(test_x, batch_size=150, verbose=0)
	metrics = model.evaluate(test_x, test_y, batch_size=150, verbose=1, sample_weight=None)

	print("Evaluation:")
	print(metrics)
	print()

	correct = []
	for i in range(len(test_pred)):
		if abs(test_y[i] - test_pred[i]) < 1e-2:
			correct.append((i, test_pred[i], test_y[i]))

	acc = len(correct)/len(test_y)

	# print("Model Predictions:")
	# print(test_pred)
	# print()
	# print("Actual Response")
	# print(test_y)
	# print()
	# view_predictions(test_pred, test_y)
	# print("Correct accuracy: {0}.  Number of correct predictions: {1}\nPredictions:".format(acc,len(correct)))
	print("Predicted {0} examples correctly.  Accuracy -> {1}".format(len(correct), acc))
	print("Predictions:")
	print(correct)
	print()

	model.save('model.h5')
	print("Model saved")
	print("===========")
	print("===========")
	print()

	cv2.imshow('First Image', test_x[0])
	input()


def train_model_exp(images, steering):
	print()
	print("=============================")
	print("=============================")
	print("TRAINING MODEL (Experimental)")
	print("=============================")
	print("=============================")

	start_time = time.time()
	shape = images.shape
	height = shape[1]
	width = shape[2]
	depth = shape[3]

	n_samples = shape[0]
	n_test = int(0.17 * n_samples)
	test_x = images[:n_test]
	test_y = steering[:n_test]
	train_x = images[n_test:]
	train_y = steering[n_test:]

	model = build_model(height, width, depth)

	# Set model training parameters
	# sgd = SGD(lr=LR, decay=DECAY)
	# model.compile(optimizer=sgd, loss='mse', metrics=['accuracy'])
	adam = Adam(lr=LR)
	# model.compile(optimizer=adam, loss='mean_squared_error', metrics=['accuracy', 'mean_absolute_error'])
	model.compile(optimizer=adam, loss='mean_absolute_error')
    # Train model
	history = model.fit(train_x, train_y, nb_epoch=EPOCHS, batch_size=BATCH, validation_split=0.2, shuffle=True)
	# Record end time
	end_time = time.time()

	print("Finished. Elapsed time: {0} mins".format((end_time - start_time)/float(60)))
	print()

	print("==============================")
	print("History:")
	print("==============================")
	for key, val in history.history.items():
		print("Metric: {0},  Final Value: {1:.3f},  Reduction:  {2:.4f}".format(key, val[EPOCHS-1], val[0] - val[EPOCHS -1]))
	print()
	# print(history.history)

	print("==============================")
	print("Predict:")
	print("==============================")
	# test_x, test_y = shuffle(images, steering, n_samples=(96))
	test_pred = model.predict(test_x, batch_size=150, verbose=0)
	metrics = model.evaluate(test_x, test_y, batch_size=150, verbose=1, sample_weight=None)

	print("Evaluation:")
	print(metrics)
	print()

	correct = []
	for i in range(len(test_pred)):
		if abs(test_y[i] - test_pred[i]) < 1e-2:
			correct.append((i, test_pred[i], test_y[i]))

	acc = len(correct)/len(test_y)

	# print("Model Predictions:")
	# print(test_pred)
	# print()
	# print("Actual Response")
	# print(test_y)
	# print()
	# view_predictions(test_pred, test_y)
	# print("Correct accuracy: {0}.  Number of correct predictions: {1}\nPredictions:".format(acc,len(correct)))
	print("Predicted {0} examples correctly.  Accuracy -> {1}".format(len(correct), acc))
	print("Predictions:")
	print(correct)
	print()

	model.save('model_exp.h5')
	print("Model saved")
	print("===========")
	print("===========")
	print()

	cv2.imshow('First Image', test_x[0])
	input("===========  Hit enter to continue  ============")


def build_model(height, width, depth):
	top_crop = int(TOP_CROP_FACTOR * height)
	bot_crop = int(BOT_CROP_FACTOR * height)

	# Build model
	model = Sequential()


	# Layer 1 (5x5): 40x80 -> 18x38
	# Layer 1 (5x5): 64x128 -> 30x62
	model.add(Cropping2D(cropping=((top_crop, bot_crop), (0, 0)), dim_ordering='tf', input_shape=(height, width, depth)))
	model.add(Convolution2D(CONV_DEPTH[0], CONV_FILTER[0], CONV_FILTER[0], dim_ordering='tf', border_mode='valid'))
	# model.add(Convolution2D(CONV_DEPTH[0], CONV_FILTER[0], CONV_FILTER[0], dim_ordering='tf', border_mode='valid', input_shape=(height, width, depth)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='same', dim_ordering='tf'))


	# Dropout
	# model.add(Dropout(DROP_PROB))


	# Layer 2 (5x5): 18x38 -> 7x17
	# Layer 2 (5x5): 30x62 -> 13x29
	model.add(Convolution2D(CONV_DEPTH[1], CONV_FILTER[1], CONV_FILTER[1], border_mode='valid', dim_ordering='tf'))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='same', dim_ordering='tf'))


	# Dropout
	# model.add(Dropout(DROP_PROB))


	# Layer 3 (3x3): 7x17 -> 3x8
	# Layer 3 (5x5): 30x62 -> 5x13
	model.add(Convolution2D(CONV_DEPTH[2], CONV_FILTER[2], CONV_FILTER[2], border_mode='valid', dim_ordering='tf'))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='same', dim_ordering='tf'))


	# Dropout
	# model.add(Dropout(DROP_PROB))


	# Layer 4 (2x2): 3x8 -> 2x7 
	# Layer 4 (3x3): 5x13 -> 3x11
	model.add(Convolution2D(CONV_DEPTH[3], CONV_FILTER[3], CONV_FILTER[3], border_mode='valid', dim_ordering='tf'))
	model.add(Activation('relu'))


	# Dropout
	# model.add(Dropout(DROP_PROB))


	# Layer 5 (2x2): 2x7 -> 1x6
	# Layer 5 (3x3): 3x11 -> 1x9
	model.add(Convolution2D(CONV_DEPTH[4], CONV_FILTER[4], CONV_FILTER[4], border_mode='valid', dim_ordering='tf'))
	model.add(Activation('relu'))


	# Dropout
	# model.add(Dropout(DROP_PROB))



	# Flatten: 64x1x6 -> 384
	# Flatten: 64x1x9 -> 576
	model.add(Flatten())


	# Layer 6
	model.add(Dense(DENSE_PARAMS[0]))
	model.add(Activation('relu'))


	# Dropout
	# model.add(Dropout(DROP_PROB))


	# Layer 7
	model.add(Dense(int(DENSE_PARAMS[1])))
	model.add(Activation('relu'))


	# Dropout
	# model.add(Dropout(DROP_PROB))


	# Layer 8
	model.add(Dense(DENSE_PARAMS[2]))
	model.add(Activation('relu'))


	# Layer 9
	model.add(Dense(DENSE_PARAMS[3]))

	return model


def view_predictions(predictions, actuals):
	print()
	print("Predicitons")
	print("(Prediction, Actual) -> diff")
	for prediction, actual in zip(predictions, actuals):
			print("({0}, {1}) -> {2}".format(prediction, actual, prediction - actual))
	print()



def main():
	# Output formatting
	print()
	# Line below is to run test code
	# test()

	# Record start time
	module_start = time.time()

	center, left, right, steering = get_img_and_steer_data_from_file()

	# Remove some of the zero data
	keep_indices = indices_of_desired_data(np.array(steering, dtype=np.float32), 0.1)
	center, left, right, steering = filter_data((center, left, right, steering), keep_indices)

	# Line below calls a function the program once the function is finished
	# look_at_data(np.array(steering, dtype=np.float32))

	# For test only
	small_center, small_left, small_right, small_steering = shuffle(center, left, right, steering, n_samples=TEST_SIZE)
	steering_center, steering_left, steering_right = augment_steering_data(small_steering)
	center_images = load_center_images(small_center)
	left_images = load_left_images(small_left)
	right_images = load_right_images(small_right)
	# steering_center, steering_left, steering_right = augment_steering_data(steering)
	# center_images = load_center_images(center)
	# left_images = load_left_images(left)
	# right_images = load_right_images(right)
	images_ctr_lft_rgt, steering_ctr_lft_rgt = stack_data(
		(center_images, left_images, right_images), 
		(steering_center, steering_left, steering_right))
	# gray_images = images_ctr_lft_rgt
	gray_images = to_grayscale(images_ctr_lft_rgt)
	# sml_images = shrink_images(gray_images, SHRINK_FACTOR)
	# norm_images = normalize(sml_images)
	norm_images = normalize(gray_images)
	aug_images, aug_steering  = create_flipped_images(norm_images, steering_ctr_lft_rgt)

	# IMAGES, STEERING = shuffle_data(aug_images, aug_steering, NUM_SAMPLES)		
	IMAGES, STEERING = shuffle_data(aug_images, aug_steering)

	# print("STEERING MEAN", np.mean(STEERING))
	# print("STEERING MIN", STEERING.min())
	# print("STEERING MAX", STEERING.max())
	# print("STEERING MODE", stats.mode(STEERING))
	# exit()

	# look_at_data(STEERING)

	shape = IMAGES.shape
	if (len(shape)==3):
		IMAGES = IMAGES.reshape(shape[0], shape[1], shape[2], 1)

	# train_model(IMAGES, STEERING)
	train_model_exp(IMAGES, STEERING)


	# Record end time
	module_end = time.time()

	print("=============================")
	print("FINISHED")
	print("=============================")
	print("Total time: {:.3f} min".format((module_end - module_start)/float(60)))
	print()
	print("FINAL IMAGES SHAPE", IMAGES.shape)
	print("FINAL STEERING SHAPE", STEERING.shape)
	# print("IMAGE_DATA:")
	# print(IMAGES[67, 20:25, 35:40, :])


if __name__ == '__main__':
	main()
