from os import listdir
from os import getcwd

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
from skimage import exposure


from keras.models import Sequential, load_model
from keras.layers.convolutional import Cropping2D, Convolution2D
from keras.layers.core import Activation, Dense, Flatten, Dropout
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import SGD, Adam


# Constants
# ===============================
# ===============================

# Directories
# ===============================
LOCAL_DIR = '/Users/cvar/selfdrivingcar/term_one/projectthree'
AWS_DIR = '/home/carnd'
DATA_DIR = None
DRIVING_FILE = 'driving_log.csv'
FOLDER = 'IMG'
VERBOSE_MODE = False

# Dataset params
# ===============================
# Shrinking factor for images
SHRINK_FACTOR = 0.4
# percent of zero turning angle datapoints to keep in the training set
P_ZERO_KEEP = 0.25
# Steering Multiple used to create new steering data
STEERING_MULT = 2
# Expected image size
TOP_CROP_FACTOR = 60.0/160.0
BOT_CROP_FACTOR = 20.0/160.0
HEIGHT = int(160*(TOP_CROP_FACTOR + BOT_CROP_FACTOR)*SHRINK_FACTOR)
WIDTH = int(320*SHRINK_FACTOR)
CROP_START = int(160 * TOP_CROP_FACTOR * SHRINK_FACTOR)
CROP_END = int((160*SHRINK_FACTOR)*(1 - BOT_CROP_FACTOR))
OUT_IMG_SIZE = (HEIGHT, WIDTH, 1)
# Logging output
print("Out size:", OUT_IMG_SIZE)
print("Crop start: {0}, end: {1}".format(CROP_START, CROP_END))


# Model params
# ===============================
EPOCHS = 10
# SAMPLES_PER_EPOCH = 4765 * 6 # Cheated to find this number that covers all data
BATCH = 100
CONV_FILTER = [5, 5, 3, 3]
CONV_DEPTH = [4, 8]
DENSE_PARAMS = [100, 50, 10, 1]
CONV_DROP = [0.5, 0.5]
DENSE_DROP = [0.4, 0.3, 0.2]
LR = 0.001


# Extra training - params for transfer learning
# ================
LRX = 0.0005
EPOCHSX = 10
in_name = 'base_lg.h5'
out_name = 'refined_lg.h5'


# Experimental model params below
# =================================

# # Successful Model
# # Model params
# # ===============================
# # ===============================
# EPOCHS = 50
# BATCH = 100
# CONV_FILTER = [5, 5]
# CONV_DEPTH = [20, 30]
# DENSE_PARAMS = [100, 50, 10, 1]
# CONV_DROP = [0.2, 0.25]
# DENSE_DROP = [0.2, 0.25, 0.2]
# LR = 0.0001
# # ===============================
# # ===============================



# # Current Model 
# # ===============================
# # ===============================
# EPOCHS = 40
# BATCH = 100
# CONV_FILTER = [5, 5, 5, 3]
# CONV_DEPTH = [20, 30, 40, 40]
# DENSE_PARAMS = [100, 50, 10, 1]
# CONV_DROP = [0.3, 0.2, 0.3, 0.2]
# DENSE_DROP = [0.3, 0.2, 0.3]
# LR = 0.0001
# # ===============================
# # ===============================


# Params for testing
# ===============================
NUM_SAMPLES = None # Value of 'None' means use the entire data set


# Steering Correction - Initialized to a dummy value 
# Steering correction is calculated later on in the process
CORRECTION = 0.1


# Training Methods
# ===============================
# ===============================

def generator_v2(img_files, steering, flip_flags):
	"""
	Python generator that takes in a list of image file paths and steering
	instructions and loads and processes batches of images

	Params
	------
	img_files: list of str -> paths to image files
	streering: list of float -> steering angles associated with each image
	flip_flags: List of boolean -> tells the generator whether or not an associated image needs to be left-right flipped
	"""

	# Get size of data set
	length = len(steering)
	print()
	print("Generator data size (length) before loading images in loop:", len(img_files), len(steering), len(flip_flags))
	print()

	# Setup batching
	batch_size = BATCH * 1
	num_batches = length/batch_size
	if(not num_batches == int(num_batches)):
		num_batches = num_batches + 1
	num_batches = int(num_batches)
	print("Num batches:", num_batches)

	# Generate data
	while 1:
		img_files, steering, flip_flags = shuffle(img_files, steering, flip_flags)
		for batch in range(num_batches):
			# Get the next batch of files and steering data
			files = img_files[batch*batch_size:(batch+1)*batch_size]
			y = steering[batch*batch_size:(batch+1)*batch_size]
			flags = flip_flags[batch*batch_size:(batch+1)*batch_size]
			size = len(files)
			x = np.empty((size, OUT_IMG_SIZE[0], OUT_IMG_SIZE[1], OUT_IMG_SIZE[2]))
			# Load and process images
			for i in range(size):
				img = load_image(files[i])
				if flags[i]:
					img = np.fliplr(img)
				x[i] = drive_preprocess(img)

			yield x, y
			# yield x, y, flip_flags


def train_model_v2(train_files, train_steering, train_flip_flags, valid_files, valid_steering, valid_flip_flags):
	"""
	Function that trains the steering model

	Params
	------
	train_files: (list of str) 		--		list of paths to images to train on
	train_steering: (list of float)	--  	list of steering angles used as response variables for each image in the training set
	train_flip_flags: (list fo boolean) -- 	list of boolean flags indicating whether an assiciated training image needs to be left-right flipped
	valid_files: (list of str) 		--  	list of paths to images to use for validation
	valid_steering:	(list of float) --		list of steering angles used as response variables for each image in the validation set
	valid_flip_flags: (list fo boolean) -- 	list of boolean flags indicating whether an assiciated validation image needs to be left-right flipped
	"""
	print()
	print("=================")
	print("=================")
	print("TRAINING V2 MODEL")
	print("=================")
	print("=================")

	# Time operation
	start_time = time.time()

	# Model input params
	height = OUT_IMG_SIZE[0]
	width = OUT_IMG_SIZE[1]
	depth = OUT_IMG_SIZE[2]

	# Build model
	model = build_model_v2(height, width, depth)
	# Finalize model with a loss function and optimizer
	adam = Adam(lr=LR)
	model.compile(optimizer=adam, loss='mean_squared_error', metrics=['mean_absolute_error'])

	# Print hyperparameters for last minute check
	print()
	print("Learning rate: ", LR)
	print("Batch size:", BATCH)
	print("Convolution layer filters:", CONV_FILTER)
	print("Convolution layer depths:", CONV_DEPTH)
	print("Dense layers:", DENSE_PARAMS)
	print("Conv Dropout", CONV_DROP)
	print("Dense Dropout", DENSE_DROP)
	print()

    # Train model
	samples_per_epoch = len(train_steering)
	nb_val_samples = len(valid_steering)
	history = model.fit_generator(
		generator=generator_v2(train_files, train_steering, train_flip_flags),
		 samples_per_epoch=samples_per_epoch, 
		 nb_epoch=EPOCHS, 
		 validation_data=generator_v2(valid_files, valid_steering, valid_flip_flags),
		 nb_val_samples=nb_val_samples)

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

	# Save model
	model.save('md.h5')
	print("=============================")
	print("=============================")
	print("Model saved as 'md.h5'")
	print("=============================")
	print("=============================")
	print()


def build_model_v2(height, width, depth):
	"""
	Function sets up the layers of a sequential Keras model

	Params
	------
	height: (int)	-- height of input image
	with: (int)		-- width of input image
	depth: (int)	-- number of channels in input image
	"""

	# Cropping was moved to the generator
	# # Calculate cropping parameters
	# top_crop = int(TOP_CROP_FACTOR * height)
	# bot_crop = int(BOT_CROP_FACTOR * height)

	# Initialize model
	model = Sequential()

	"""
	Layer 1:
	Input: 32x128x1 -> 5x5x4 Convolution Filter  with 2x2 strides -> Output:  14x62x4
	Activation -> 14x62x4
	"""
	model.add(Convolution2D(
		CONV_DEPTH[0], CONV_FILTER[0], CONV_FILTER[0], 
		dim_ordering='tf', border_mode='valid', activation='relu', 
		subsample=(2,2), input_shape=(height, width, depth)))
	model.add(Dropout(CONV_DROP[0]))

	"""
	Layer 2:  Input: 14x62x4 -> 5x5x8 Convolution Filter  with 2x2 strides -> Output:  5x29x8
	Activation -> 5x29x8
	"""
	model.add(Convolution2D(
		CONV_DEPTH[1], CONV_FILTER[1], CONV_FILTER[1], 
		dim_ordering='tf', border_mode='valid', activation='relu', 
		subsample=(2,2)))
	model.add(Dropout(CONV_DROP[1]))

	# Flatten:  Input: 5x29x8 -> Flatten -> Output:  1160
	model.add(Flatten())

	# Layer 3:  Input: 1160 -> Dense(100)
	model.add(Dense(DENSE_PARAMS[0], activation='relu'))
	model.add(Dropout(DENSE_DROP[0]))

	# Layer 4:  Input: 100 -> Dense(50)
	model.add(Dense(DENSE_PARAMS[1], activation='relu'))
	model.add(Dropout(DENSE_DROP[1]))

	# Layer 5:  Input: 50 -> Dense(10)
	model.add(Dense(DENSE_PARAMS[2], activation='relu'))
	model.add(Dropout(DENSE_DROP[2]))

	# Layer 6:  Input: 10 -> Dense(1)
	model.add(Dense(DENSE_PARAMS[3]))

	return model


def organize_data():
	"""
	Function:
	1.  Opens the csv log file
	2.  Reads in all of the file paths and steering info
	3.  Throws away 75% of the zero steering angle data
	4.  Creates steering data for images in the left and right cameras
	5.  duplicates the steering and camera files data to be used later as inverted images
	6.  Creates an array of boolean flags to tell the generator which images need to be flipped
	7.  Stacks all image, steering, and flipped image flags data into three arrays
	8.  Randomly selects 20% of the data for model validaiton use
	9.  Returns the two data sets: training and validation, as six seperate arrays (image files, steering info, and boolean flip flags)
	"""
	# Output log formatting
	print()
	print("Organizing Data")
	print()

	# Record start time
	module_start = time.time()

	# Get data from files
	center, left, right, steering = get_data()

	# Set the steering correction angle for later
	steering = np.array(steering, dtype=np.float32)
	avg_steer = np.mean(np.absolute(steering))
	global CORRECTION
	CORRECTION = avg_steer * STEERING_MULT

	# Remove some of the zero data
	keep_indices = indices_of_desired_data(steering, P_ZERO_KEEP)
	center, left, right, steering = filter_data((center, left, right, steering), keep_indices)

	# For test only
	if NUM_SAMPLES is not None:
		print("NOT USING ENTIRE DATA SET. Sample size:", NUM_SAMPLES)
		center, left, right, steering = shuffle(center, left, right, steering, n_samples=NUM_SAMPLES)

	# Setup info for flipping photos later in the pipeline
	size = 3 * len(steering)
	reg_img = np.zeros(size, dtype=bool)
	flip_img = np.ones(size, dtype=bool)

	# Stack data into two arrays
	center_steer, left_steer, right_steer = augment_steering_data(steering)
	steering = np.concatenate((center_steer, left_steer, right_steer, -center_steer, -left_steer, -right_steer), axis=0)
	img_files = np.concatenate((center, left, right, center, left, right), axis=0)
	flip_flags = np.concatenate((reg_img, flip_img), axis=0)

	# Shuffle data
	img_files, steering, flip_flags = shuffle(img_files, steering, flip_flags)

	# Partition data
	num_samples = len(steering)
	valid_factor = 0.2
	valid_count = int(valid_factor * num_samples)

	# Validation data
	valid_files = img_files[0:valid_count]
	valid_steering = steering[0:valid_count]
	valid_flip_flags = flip_flags[0:valid_count]

	# Training data
	train_files = img_files[valid_count:]
	train_steering = steering[valid_count:]
	train_flip_flags = flip_flags[valid_count:]

	print("Sample count -> Training: {0}  Validation: {1}".format(len(train_steering), len(valid_steering)))

	return train_files, train_steering, train_flip_flags, valid_files, valid_steering, valid_flip_flags


def train_gpu():
	"""
	Function serves as a 'main' function.  First function called from the command line. 
	Function times the entire training process and calls the training function 'train_model_v2'
	"""
	# Output formatting
	print()
	print("Begin training")
	print()

	# Record start time
	module_start = time.time()

	# Train model
	train_model_v2(*organize_data())

	# Record end time
	module_end = time.time()

	print("=============================")
	print("FINISHED")
	print("=============================")
	print("Total time: {:.3f} min".format((module_end - module_start)/float(60)))
	print()


def drive_preprocess(image):
	"""
	Function processes an image and prepares it for the keras model

	1.  Shrinks the images to 40% of the original
	2.  Converts to grayscale
	3.  Crops the top and bottom of the image (only keep portion of photo that encloses the road)
	4.  Normalizes the distribution of the image data from [0,255] to [-0.5,+0.5]

	Params:
	-------
	image: (ndarray)	-- image to process
	"""

	image = cv2.resize(image, None, fx=SHRINK_FACTOR, fy=SHRINK_FACTOR, interpolation = cv2.INTER_AREA)
	image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).reshape(64,128,1)
	crop = image[CROP_START:CROP_END,:,:]
	ret_img = (crop/255) - 0.5

	return ret_img


# Helper Functions
#====================================
#====================================


def get_parent_dir():
	cwd = getcwd()
	dir_array = cwd.split('/')
	dir_array.pop()

	return '/'.join(dir_array)


def get_data_dir():
	return get_parent_dir() + '/data'


def get_driving_path():
	return get_data_dir() + '/driving_log.csv'


def using_gpu():
	return get_parent_dir() == AWS_DIR


def get_data():
	"""
	Function reads in data from the driving log
	"""
	steering = []
	center = []
	left = []
	right = []

	# Logging output
	if VERBOSE_MODE:
		print("=============================")
		print("Reading in file")
		print("=============================")

	# Time operation
	start_time = time.time()

	# Read in file and save data
	# =============================
	with open(get_driving_path()) as csv_file:
		reader = csv.reader(csv_file)
		next(reader)
		for row in reader:
			center.append(row[0])
			left.append(row[1])
			right.append(row[2])
			steering.append(row[3])
	end_time = time.time()

	# Logging output
	if VERBOSE_MODE:
		print("Finished. Elapsed time: {0} secs".format(end_time - start_time))
		print()

	return center, left, right, steering


def indices_of_desired_data(steering, p_zero_keep):
	"""
	Function to throw away a percentage of the zero angle steering data. 
	Function returns indices of desired training data 

	Params:
	-------
	steering (array-like - floats)	-- steering data to filter
	p_zero_keep (float)				-- percentage of zero angle data to keep 
	"""

	# Log output
	if VERBOSE_MODE:
		print("=============================")
		print("Finding indices to keep")
		print("=============================")

	# Time operationa
	start_time = time.time()

	# Iterate through data and find zero steering angles
	reg_indices = []
	zero_indices = []
	for i, steer in enumerate(steering):
		if abs(steer) < 0.001:
			zero_indices.append(i)
		else:
			reg_indices.append(i)

	# Log output
	size_reg = len(reg_indices)
	size_zero = len(zero_indices)
	size_total = size_reg + size_zero
	print("Reg: {0} -> {1:.2f}".format(size_reg, size_reg/size_total))
	print("Zero: {0} -> {1:.2f}".format(size_zero, size_zero/size_total))
	print()

	# Randomly select zero steering angle data to keep
	n_zeros = int(p_zero_keep * size_zero)
	zero_keep_indices = shuffle(zero_indices, n_samples=n_zeros)
	# Make a list of indices of all steering data
	keep_indices = np.concatenate((reg_indices, zero_keep_indices))
	print("Percentage of zero steering data to keep", p_zero_keep)
	print("Total Keep:", len(keep_indices))

	# Record end time
	end_time = time.time()

	# Log output
	if VERBOSE_MODE:
		print("Finished. Elapsed time: {0} secs".format(end_time - start_time))

	return keep_indices


def filter_data(lists, indices):
	"""
	Function filters data from a list of lists based on the indices provided

	Params:
	-------
	lists (list or tuple of array-like objects)	--  Lists to operate on
	indices (array-like object of type int)		--  indices of data to select
	"""

	# Log output
	if VERBOSE_MODE:
		print("=============================")
		print("Filtering lists")
		print("=============================")

	# Time operation
	start_time = time.time()

	# Create object for return data
	out_lists = []
	num_lists = len(lists)
	for i in range (num_lists):
		out_lists.append([])
	# For each index in indices, iterate through input lists and retreive and store desired data
	for index in indices:
		for j in range(num_lists):
			out_lists[j].append(lists[j][index])

	# Record finish time
	end_time = time.time()

	# Log output
	if VERBOSE_MODE:
		print("Finished. Elapsed time: {0} secs".format(end_time - start_time))

	return out_lists



def augment_steering_data(steering):
	"""
	Function creates new steering data from the provided steering data to use for images in the left and right cameras.
	Function takes the average of the magnitude of the steering data set and uses this average as the base metric for
	the 'correction' value.  The correction value is used as a modifier that tells the model that images that are 
	offset a little further left(right) should have the steering angle increased(decreased) by the 'correction' value.

	Params:
	-------
	steering (array-like of floats)	-- steering data to use as a base to build new steering data
	"""

	# Log output
	if VERBOSE_MODE:
		print("=============================")
		print("Creating augmented steering data")
		print("=============================")

	# Time operation
	start_time = time.time()

	# Log output
	print("Creating augmented steering data with correction: {0}    (Steering Multiple = {1})".format(CORRECTION, STEERING_MULT))

	# Log output
	if VERBOSE_MODE:
		print("AVG_STEER:", AVG_STEER)
		print("CORRECTION:", CORRECTION)

	# make augmented data
	steering_center = np.array(steering, dtype=np.float32)
	steering_left = steering_center.copy() + CORRECTION
	steering_right = steering_center.copy() - CORRECTION

	# Record finish time
	end_time = time.time()

	# Log output
	if VERBOSE_MODE:
		print("Finished. Elapsed time: {0} secs".format(end_time - start_time))
		print()

	return steering_center, steering_left, steering_right


def load_images(raw_paths):
	"""
	Function loads images from a list of file paths into ndarrays

	Params:
	-------
	raw_paths (str)	--  Paths to the image files
	"""

	# Log output
	if(VERBOSE_MODE):
		print("=============================")
		print("Processing images")
		print("=============================")

	# Time operation
	start_time = time.time()

	# Iterate through each path in list, load image, and store in an array
	images = []
	DATA_DIR = get_data_dir()
	for i, raw_path in enumerate(raw_paths):
		start = raw_path.find(FOLDER)
		suffix = raw_path[start:]
		img_path = DATA_DIR + '/' + suffix
		image = cv2.imread(img_path)
		images.append(image)
		if(image is None):
			print("None type for index: {0}, path: {1}".format(i, img_path))
			exit()

	# Record end time
	end_time = time.time()

	# Convert from list to ndarray
	images = np.asarray(images)

	# Log output
	if(VERBOSE_MODE):
		print("Finished. Elapsed time: {0} secs".format(end_time - start_time))
		print("Shape & type left images:", images.shape, type(images))
		print()

	return images


def load_image(raw_path):
	"""
	Function loads images from a list of file paths into ndarrays

	Params:
	-------
	raw_paths (str)	--  Paths to the image files
	"""

	# Iterate through each path in list, load image, and store in an array
	start = raw_path.find(FOLDER)
	suffix = raw_path[start:]
	img_path = DATA_DIR + '/' + suffix
	return cv2.imread(img_path)



def stack_data(img_tuple, steer_tuple):
	"""
	Consolidates two tuples of arrays into two ndarrays

	Params:
	-------
	img_tuple (tuple of array-like objects)		--  Tuple to stack into an ndarray
	steer_tuple (tuple of array-like objects)	--  Tuple to stack into an ndarray
	"""

	# Log output
	if VERBOSE_MODE:
		print("=============================")
		print("Stacking data")
		print("=============================")

	# Time operation
	start_time = time.time()

	# Stack data
	img_array = np.concatenate(img_tuple, axis=0)
	steer_array = np.concatenate(steer_tuple, axis=0)

	# Record finish time
	end_time = time.time()

	# Log output
	if VERBOSE_MODE:
		print("Finished. Elapsed time: {0} secs".format(end_time - start_time))
		print("shape of stacked images:", images_ctr_lft_rgt.shape)
		print("shape of stacked steering:", steering_ctr_lft_rgt.shape)
		print()

	return img_array, steer_array


def create_flipped_images(images, steering):
	"""
	Create augmented data by flipping the image and the steering angle.

	Params:
	-------
	images (array-like of images)	--  images to perform a left right flip operation on
	steering (array-like of floats)	--  steering data to invert
	"""

	# Log output
	if VERBOSE_MODE:
		print("=============================")
		print("Creating augmented data")
		print("=============================")

	# Record operation
	start_time = time.time()


	# Iterate through image and steering data, flip/invert, and store.
	image_list = []
	steering_list = []
	for image, steer in zip(images, steering):
		image2 = np.fliplr(image)
		steer2 = steer * -1.0
		# Append regular
		image_list.append(image)
		steering_list.append(steer)
		# Append flipped
		image_list.append(image2)
		steering_list.append(steer2)

	# Convert lists to ndarrays
	IMAGES = np.asarray(image_list)
	STEERING = np.asarray(steering_list)

	# Record finish time
	end_time = time.time()

	# Log output
	if VERBOSE_MODE:
		print("Finished. Elapsed time: {0} secs".format(end_time - start_time))
		print("shape IMAGES:", IMAGES.shape)
		print("shape STEERING:", STEERING.shape)
		print()

	return IMAGES, STEERING


def norm_img(image):
	a = -0.5
	b = 0.5
	grayscale_min = 0
	grayscale_max = 255
	return a + ( ( (image - grayscale_min)*(b - a) )/( grayscale_max - grayscale_min ) )


def initializations():
	# Initialize directory
	global DATA_DIR
	DATA_DIR = get_data_dir()


def view_images():
	"""
	Test function used for debugging
	"""
	# x, y = (next(generator()))
	train_files, train_steering, train_flip_flags, valid_files, valid_steering, valid_flip_flags = organize_data()

	n_samples = 6
	x, y = shuffle(train_files, train_steering, n_samples = n_samples)
	i = [i for i in range(n_samples)]
	DATA_DIR = get_data_dir()
	for raw_path, steer in zip(x, y):
		start = raw_path.find(FOLDER)
		suffix = raw_path[start:]
		img_path = DATA_DIR + '/' + suffix
		img = cv2.imread(img_path)
		cv2.imshow('Instruction {0:.4f}'.format(steer), img)
		cv2.waitKey(6000)	
	# cv2.imshow('Generator test image', x[0])
	# cv2.waitKey(3000)
	input("================  Continue? ====================")


def test():
	"""
	Test function used for debugging
	"""
	# x, y = (next(generator()))
	train_files, train_steering, train_flip_flags, valid_files, valid_steering, valid_flip_flags = organize_data()
	start_time = time.time()
	x, y, flip_flags = (next(generator_v2(valid_files, valid_steering, valid_flip_flags)))
	end_time = time.time()
	print("Generator time:", end_time - start_time)
	print("Generator delivered list of size:", len(x))
	n_samples = None
	timer = 3000
	x, y, flip_flags = shuffle(x, y, flip_flags, n_samples=n_samples)
	for img, steer, flip in zip(x,y, flip_flags):
		print("shape", img.shape)
		cv2.imshow('{0:.4f} {1}'.format(steer, flip), img)
		cv2.waitKey(timer)
		cv2.destroyAllWindows()
	input("================  Continue? ====================")


def gen_test_flip():
	"""
	Test function used for debugging
	"""
	train_files, train_steering, train_flip_flags, valid_files, valid_steering, valid_flip_flags = organize_data()

	for i in range(10):
		# x, y, flip_flags = (next(generator_v2(valid_files, valid_steering, valid_flip_flags)))
		x, y, flip_flags = (next(generator_v2(train_files, train_steering, train_flip_flags)))
		print("Gen len", len(x), len(y), len(flip_flags))
		x, y, flip_flags = shuffle(x, y, flip_flags, n_samples=4)
		for j, img in enumerate(x):
			# cv2.imshow("{0}, {1}: {2}: {3:.3f}".format(i,j,flip_flags[j], y[j]), img)
			cv2.imshow("{0}, {1}: {2}: {3:.3f}".format(i,j,'', y[j]), img)
			cv2.waitKey(2000)
			cv2.destroyAllWindows()

def gen_test():
	"""
	Test function used for debugging
	"""
	train_files, train_steering, train_flip_flags, valid_files, valid_steering, valid_flip_flags = organize_data()

	for i in range(2):
		# x, y, flip_flags = (next(generator_v2(valid_files, valid_steering, valid_flip_flags)))
		start_time = time.time()
		x, y = (next(generator_v2(train_files, train_steering, train_flip_flags)))
		end_time = time.time()
		print("Generator time:", end_time - start_time)
		print("Gen len", len(x), len(y))
		x, y = shuffle(x, y, n_samples=4)
		for j, img in enumerate(x):
			# cv2.imshow("{0}, {1}: {2}: {3:.3f}".format(i,j,flip_flags[j], y[j]), img)
			print(img)
			print(img.shape)
			cv2.imshow("{0}, {1}: {2}: {3:.3f}".format(i,j,'', y[j]), img)
			cv2.waitKey(8000)
			cv2.destroyAllWindows()


def extra_training(train_files, train_steering, train_flip_flags, valid_files, valid_steering, valid_flip_flags):
	"""
	Function that opens an already trained model and preforms more training on it.

	Params:
	-------
	train_files: (list of str) 		--		list of paths to images to train on
	train_steering: (list of float)	--  	list of steering angles used as response variables for each image in the training set
	train_flip_flags: (list fo boolean) -- 	list of boolean flags indicating whether an assiciated training image needs to be left-right flipped
	valid_files: (list of str) 		--  	list of paths to images to use for validation
	valid_steering:	(list of float) --		list of steering angles used as response variables for each image in the validation set
	valid_flip_flags: (list fo boolean) -- 	list of boolean flags indicating whether an assiciated validation image needs to be left-right flipped
	"""
	print()
	print("==============================")
	print("==============================")
	print("TRANSFER LEARNIN ON BASE MODEL")
	print("==============================")
	print("==============================")

	start_time = time.time()

	print()
	print("Input model name:", in_name)
	print()
	model = load_model(in_name)

	height = OUT_IMG_SIZE[0]
	width = OUT_IMG_SIZE[1]
	depth = OUT_IMG_SIZE[2]

	adam = Adam(lr=LRX)
	model.compile(optimizer=adam, loss='mean_squared_error', metrics=['mean_absolute_error'])

	# Print hyperparameters for last minute check
	print()
	print("Learning rate: ", LRX)
	print("Batch size:", BATCH)
	print("Convolution layer filters:", CONV_FILTER)
	print("Convolution layer depths:", CONV_DEPTH)
	print("Dense layers:", DENSE_PARAMS)
	print("Conv Dropout", CONV_DROP)
	print("Dense Dropout", DENSE_DROP)
	print()

    # Train model
	samples_per_epoch = len(train_steering)
	nb_val_samples = len(valid_steering)
	history = model.fit_generator(
		generator=generator_v2(train_files, train_steering, train_flip_flags),
		 samples_per_epoch=samples_per_epoch, 
		 nb_epoch=EPOCHSX, 
		 validation_data=generator_v2(valid_files, valid_steering, valid_flip_flags),
		 nb_val_samples=nb_val_samples)


	# Record end time
	end_time = time.time()

	# save model
	model.save(out_name)

	# Log output
	print()
	print("=================================")
	print("=================================")
	print("Model Saved as:", out_name)
	print("=================================")
	print("=================================")

	print()
	print("Finished. Elapsed time: {0} mins".format((end_time - start_time)/float(60)))
	print()






if __name__ == '__main__':
	initializations()

	# Initialize and train a model
	train_gpu()

	# # Load an already trained model and train it further
	# extra_training(*organize_data())

	# # debugging function	
	# test()

	# # debugging function
	# gen_test()



