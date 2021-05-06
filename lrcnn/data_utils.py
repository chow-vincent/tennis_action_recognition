"""
Functions for preparing, processing, and manipulating data.
This file was adapted from https://github.com/harvitronix/five-video-classification-methods
to work with our dataset and approach to training an LRCNN model.

"""

from keras.applications.inception_v3 import preprocess_input #, InceptionV3
from keras.preprocessing import image
from keras.utils import to_categorical

import numpy as np
import cv2
import random
import csv

import time
import os
import os.path

import threading
import operator


# -------- FUNCTIONS FOR SAFE MULTI-THREADING ----------- #
class threadsafe_iterator:
	def __init__(self, iterator):
		self.iterator = iterator
		self.lock = threading.Lock()

	def __iter__(self):
		return self

	# def __next__(self):
	def next(self):  # NOTE: Python 2.x uses next(), Python 3.x uses __next__()
		with self.lock:
			return next(self.iterator)

def threadsafe_generator(func):
	"""Decorator"""
	def gen(*a, **kw):
		return threadsafe_iterator(func(*a, **kw))
	return gen


class DataSet():

	def __init__(self, cnn_model, 
		seq_length=16, image_shape=(227, 227, 3)):
		""" Constructor for DataSet() class
		cnn_model = keras Model, used for generating sequences
		seq_length = (int) the number of frames to consider
		class_limit = (int) number of classes to limit the data to.
		"""

		# InceptionV3 model will be used to extract features from frames
		self.cnn_model = cnn_model

		# length of each video in the dataset (default downsampled to 16 frames)
		self.seq_length = seq_length

		# directory containing saved .npy sequences
		self.sequence_path = os.path.join('data', 'sequences')

		# obtain the dataset info from data_file.csv
		self.data = self.get_data()

		# obtain info on the classes from self.data
		self.classes = self.get_classes()

		# FOR NOW, THE FOLLOWING ARE NOT BEING USED
		self.image_shape = image_shape  # for potentially changing image sizes down the road

		# self.class_limit = class_limit  # for potentially limiting number of classes


	@staticmethod
	def get_data():
		"""Load our data from file."""
		with open(os.path.join('data', 'data_file.csv'), 'r') as fin:
			reader = csv.reader(fin)
			data = list(reader)
		return data


	def get_classes(self):
		"""
		This function creates a list of classes from data_file.csv.
		NOTE: no class limit applied here, b/c we only have 12 classes to work with.
		"""
		classes = []
		for item in self.data:
			if item[1] not in classes:
				classes.append(item[1])

		classes = sorted(classes)  # sorted, but prob not necessary

		return classes


	def get_class_one_hot(self, class_label):
		"""
		This function returns a one-hot vector corresponding to the class label.
		"""
		label = self.classes.index(class_label)

		label_hot = to_categorical(label, len(self.classes))

		assert len(label_hot) == len(self.classes)

		return label_hot


	# train and test are lists, with each element a line in csv file
	def split_dataset(self):
		train = []
		validation = []
		test = []
		for item in self.data:
			if item[0] == 'train':
				train.append(item)
			elif item[0] == 'validation':
				validation.append(item)
			else:
				test.append(item)
		return train, validation, test


	def get_frames_for_sample(self, sample):
		"""
		This function, used in extract_seq_features(), obtains a list of
		frames from a given sample video. Each frame is a N x M x 3 array.
		"""

		# e.g. "data/train/backhand/p1_backhand_s1.avi"
		# path = os.path.join('data', sample[0], sample[1], sample[2] + ".avi")

		# e.g. "VIDEO_RGB/backhand/p1_backhand_s1.avi"
		path = os.path.join('VIDEO_RGB', sample[1], sample[2] + '.avi')
		
		vidcap = cv2.VideoCapture(path)
		
		frames = []
		
		# extract frames
		while True:
			success, image = vidcap.read()
			if not success:
				break
				
			img_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
			
			frames.append(img_RGB)
			
			
		# downsample if desired and necessary
		if self.seq_length < len(frames):
			skip = len(frames) // self.seq_length
			frames = [frames[i] for i in range(0, len(frames), skip)]
			frames = frames[:self.seq_length]
		
		return frames


	def extract_seq_features(self, sample):
		"""
		This function, used in get_extracted_sequence(), returns 
		a sequence if not already on disc, and saves as a .npy file.
		"""
		
		path = os.path.join('data', 'sequences', sample[1], sample[2] + '-' + str(self.seq_length) + \
		'-features')
		
		frames = self.get_frames_for_sample(sample)

		sequence = []
		for img in frames:
			x = image.img_to_array(img)
			x = np.expand_dims(x, axis=0)
			x = preprocess_input(x)
			features = self.cnn_model.predict(x)  # for some reason not on graph?

			sequence.append(features[0])  # only take first dimension

		np.save(path, sequence)
		
		return sequence


	def get_extracted_sequence(self, sample):
		"""
		This function is used in frame_generator(). Returns a sequence
		(I believe this is a list) from a .npy file stored on disc
		or creates it on the fly if not and saves as .npy for future use.

		Each sequence is a list, with each element containing the feature
		vector for each frame.
		"""

		filename = sample[2]
		
		path = os.path.join(self.sequence_path, sample[1], filename + '-' + str(self.seq_length) + \
			'-features.npy')
		
		# return saved numpy sequence
		if os.path.isfile(path):
			return np.load(path)
		
		# else we generate the numpy sequence now (saved on disc for future use)
		else:
			return self.extract_seq_features(sample)


	def get_frames_by_filename(self, filename):

		sample = None
		for row in self.data:
			if row[2] == filename:
				sample = row
				break
		if sample is None:
			raise ValueError("Could not find sample")

		sequence = self.get_extracted_sequence(sample)

		return sequence


	def print_class_from_prediction(self, predictions):

		label_predictions = {}

		for i, label in enumerate(self.classes):
			label_predictions[label] = predictions[i]

		sorted_lps = sorted(label_predictions.items(), key=operator.itemgetter(1), reverse=True)

		for i, class_prediction in enumerate(sorted_lps):
			print("%s: %.2f" % (class_prediction[0], class_prediction[1]))


	# @threadsafe_generator
	def frame_generator(self, batch_size, train_validate):
		"""
		This function creates a generator that we will use during training.
		"""

		# random.seed(1)  # for reproducibility in experiments?
		
		# not using actual test data, using validation data during training
		train, validation, _ = self.split_dataset()
		data = train if train_validate == 'train' else validation
		
		print("Creating %s generator with %d samples.\n" % (train_validate, len(data)))

		if train_validate == 'train':
			while 1:
				X, y = [], []

				# generate samples for batch (of size batch_size)
				for _ in range(batch_size):

					sequence = None

					# randomly pick a datapoint (what if we have already picked point in batch?)
					sample = random.choice(data)

					sequence = self.get_extracted_sequence(sample)

					if sequence is None:
						raise ValueError("Unable to find sequence!")

					X.append(sequence)

					class_label = sample[1]  # from csv line
					y.append(self.get_class_one_hot(class_label))

				# yield batches as necessary to fit_generator() fxn
				yield np.array(X), np.array(y)
		else:
			while 1:
				X, y = [], []
				for i in range(len(data)):

					sequence = None

					sample = data[i]

					sequence = self.get_extracted_sequence(sample)

					if sequence is None:
						raise ValueError("Unable to find sequence!")

					X.append(sequence)

					class_label = sample[1]  # from csv line
					y.append(self.get_class_one_hot(class_label))

				# print "yielding entire validation set"
				yield np.array(X), np.array(y)


	def generate_data(self, train_validate_test):
		"""
		This function generates desired training data
		"""
		train, validation, test = self.split_dataset()
		if train_validate_test == 'train':
			data = train
		elif train_validate_test == 'validation':
			data = validation
		elif train_validate_test == 'test':
			data = test

		X, y = [], []

		# loop over list of validation samples, and create sequences
		for sample in data:

			sequence = None

			sequence = self.get_extracted_sequence(sample)

			if sequence is None:
				raise ValueError("Unable to find sequence!")

			X.append(sequence)

			class_label = sample[1]  # from csv line
			y.append(self.get_class_one_hot(class_label))

		# print "yielding entire validation set"
		return np.array(X), np.array(y)

