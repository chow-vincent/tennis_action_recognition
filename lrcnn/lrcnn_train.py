"""
Train the LRCNN model. Parameters are set in params.json file in the
relevant directory under "/experiments".
"""

from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.callbacks import TensorBoard, ModelCheckpoint, CSVLogger
from keras.optimizers import Adam
from keras.models import Model, Sequential, load_model
from keras.layers import Input, LSTM, Dense, TimeDistributed, Lambda, Dropout
from keras import backend as K
from keras import regularizers
from data_utils import DataSet

import numpy as np
import os
import os.path
import time

from utils import Params
from utils import set_logger
import argparse
import logging

# import seaborn as sns
import matplotlib.pyplot as plt
import pickle


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/base_model',
										help="Directory containing params.json")

# --- RNN MODEL --- #
def lstm(num_features=2048, hidden_units=256, dense_units=256, reg=1e-1, dropout_rate=1e-1, seq_length=16, num_classes=6):
		
	# hidden_units: dimension of cell
	# dense_units: number of neurons in fully connected layer above LSTM
	# reg: regularization for LSTM and dense layer
	# - currently adding L2 regularization for RNN connections, and for inputs to dense layer
	
	model = Sequential()
	
	# return_sequences flag sets whether hidden state returned for each time step
	# NOTE: set return_sequences=True if using TimeDistributed, else False


	# LSTM layer (dropout)
	model.add(Dropout(dropout_rate, input_shape=(seq_length, num_features)))  # input to LSTM
	model.add(LSTM(hidden_units, return_sequences=True))
	
	# --- AVERAGE LSTM OUTPUTS --- #
	
	# dropout between LSTM and softmax
	model.add(TimeDistributed(Dropout(dropout_rate)))

	# commenting out additional FC layer for now
	# model.add(TimeDistributed(Dense(dense_units)))
	
	# apply softmax
	model.add(TimeDistributed(Dense(num_classes, activation="softmax")))
	
	# average outputs
	average_layer = Lambda(function=lambda x: K.mean(x, axis=1))
	model.add(average_layer)
	
	# --- ONLY TAKE LAST LSTM OUTPUT --- #
	# model.add(Dense(dense_units))
	# model.add(Dense(num_classes, activation="softmax"))
	
	return model


# --- TRAINING FUNCTION --- #
def train(model_dir, cnn_model, saved_model=None, 
			learning_rate = 1e-5, decay=1e-6, 
			train_size = 0.8, seq_length=16,
			hidden_units=256, dense_units=256, reg=1e-1, dropout_rate=1e-1,
			num_classes=6, batch_size=16, nb_epoch=100, 
			image_shape=None):


	# ---- CALL BACK FUNCTIONS FOR FIT_GENERATOR() ---- #
	checkpoints_dir = os.path.join(model_dir, 'checkpoints')
	if not os.path.exists(checkpoints_dir):
		os.makedirs(checkpoints_dir)

	checkpointer = ModelCheckpoint(
			filepath=os.path.join(checkpoints_dir, 'lstm_weights.{epoch:004d}-{val_loss:.3f}.hdf5'),
			verbose=1, save_best_only=False, period=50)
	
	# tensorboard info
	tb = TensorBoard(log_dir=model_dir)

	# ------------------------------------------------- # 


	# PREPARE DATASET
	dataset = DataSet(cnn_model, seq_length)
	
	# steps_per_epoch = number of batches in one epoch
	steps_per_epoch = (len(dataset.data) * train_size) // batch_size

	# create train and validation generators
	generator = dataset.frame_generator(batch_size, 'train')
	# val_generator = dataset.frame_generator(batch_size, 'validation') # use all validation data each time?
	(X_val, y_val) = dataset.generate_data('validation')

	# load or create model
	if saved_model:
		rnn_model = load_model(saved_model)
	else:
		rnn_model = lstm(hidden_units=hidden_units, dense_units=dense_units, 
						reg=reg, dropout_rate=dropout_rate,
						seq_length=seq_length, num_classes=num_classes)
	
	# setup optimizer: ADAM algorithm
	optimizer = Adam(lr=learning_rate, decay=decay)
	
	# metrics for judging performance of model
	metrics = ['categorical_accuracy'] # ['accuracy']  # if using 'top_k_categorical_accuracy', must specify k
	
	rnn_model.compile(loss='categorical_crossentropy', optimizer=optimizer,
			metrics=metrics)

	print(rnn_model.summary())

	# use fit generator to generate data on the fly
	history = rnn_model.fit_generator(generator=generator,
									steps_per_epoch=steps_per_epoch,
									epochs=nb_epoch,
									verbose=1,
									callbacks=[tb, checkpointer],
									validation_data=(X_val, y_val),
									validation_steps=1)  # using all validation data for better metrics

	return history


if __name__ == '__main__':

	args = parser.parse_args()
	json_path = os.path.join(args.model_dir, 'params.json')
	assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
	params = Params(json_path)

	# ----- DUMMY MODEL TO PASS IN AS CNN_MODEL ---- #
	#       (feature extraction not performed in this script)

	a = Input(shape=(1,))
	b = Dense(1)(a)
	model = Model(inputs=a, outputs=b)

	cnn_model = Model(inputs=a, outputs=b)


	# ---- LOAD PARAMETERS FOR TRAINING ---- #

	if params.saved_model == "None":
		saved_model = None
	else:
		saved_model = params.saved_model

	# --- hyperparameters --- #
	learning_rate = params.learning_rate
	decay = params.decay

	hidden_units = params.hidden_units
	dense_units = params.dense_units

	reg = params.reg
	dropout_rate = params.dropout_rate

	batch_size = params.batch_size
	nb_epoch = params.nb_epoch

	# --- other parameters --- #
	train_size = params.train_size
	num_classes = params.num_classes
	seq_length = params.seq_length

	# --- EXECUTE TRAINING --- #
	history = train(args.model_dir, cnn_model, saved_model=saved_model, 
						learning_rate = learning_rate, decay = decay, 
						train_size = train_size, seq_length = seq_length,
						hidden_units = hidden_units, dense_units = dense_units, reg=reg,
						num_classes = num_classes,
						batch_size = batch_size, nb_epoch = nb_epoch)


	print("\nCompleted training! \n")

	# --- SAVE HISTORY AS PICKLE FILE --- #
	save_path = os.path.join(args.model_dir, 'train_history.pkl')

	with open(save_path, 'wb') as save_file:
		pickle.dump(history.history, save_file, protocol=2)  # make sure python 2.7 can read
