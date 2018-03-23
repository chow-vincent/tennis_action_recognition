"""
Evaluate saved model on test data. 
Perform misclassification analysis.
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

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


# --- RNN model --- #
def lstm(num_features=2048, hidden_units=256, dense_units=256, reg=1e-1, dropout_rate=1e-1, seq_length=16, num_classes=6):

    # hidden_units: dimension of cell
    # dense_units: number of neurons in fully connected layer above LSTM
    # reg: regularization for LSTM and dense layer
    # - currently adding L2 regularization for RNN connections, and for inputs to dense layer

    model = Sequential()

    # return_sequences flag sets whether hidden state returned for each time step
    # NOTE: set return_sequences=True if using TimeDistributed, else False

    # LSTM layer (L2)
    #     model.add(LSTM(hidden_units, input_shape=(seq_length, num_features), 
    #                    return_sequences=True,
    #                    kernel_regularizer=regularizers.l2(reg),
    #                    recurrent_regularizer=regularizers.l2(reg)))

    # LSTM layer (dropout)
    model.add(Dropout(dropout_rate, input_shape=(seq_length, num_features)))  # input to LSTM
    model.add(LSTM(hidden_units, return_sequences=True))

    # --- AVERAGE LSTM OUTPUTS --- #

    # linear activation layer (L2 regularization)
    #     model.add(TimeDistributed(Dense(dense_units, 
    #                                     kernel_regularizer=regularizers.l2(reg))))

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
    # model.add(Dense(dense_units, kernel_regularizer=regularizers.l2(reg)))
    # model.add(Dense(num_classes, activation="softmax", kernel_regularizer=regularizers.l2(reg)))

    return model


if __name__ == "__main__":

    # ------ REPLICATE PARAMETERS USED IN EXPERIMENT ------ #

    # ---- HYPER-PARAMETERS TO BE SET ---- #

    learning_rate= 1e-3 # 1e-4
    decay=0.0

    hidden_units = 128
    dense_units = 128
    reg = 0.0           # L2 regularization
    dropout_rate = 0.3  # dropout regularization
    batch_size = 128
    nb_epoch = 300 # 100


    # ---- OTHER PARAMETERS ---- #
    train_size = 0.8  # proportion of dataset that is training
    saved_model = None  # None, or pass in weights file
    # saved_model = "data/checkpoints/lstm_weights.0026-0.239.hdf5"

    num_classes = 6
    seq_length = 16    # essentially number of frames


    # --- LOAD DUMMY CNN MODEL --- #
    a = Input(shape=(1,))
    b = Dense(1)(a)
    model = Model(inputs=a, outputs=b)

    cnn_model = Model(inputs=a, outputs=b)


    # ------ EVALUATE MODEL ------ #
    # have to reinstantiate model to load weights properly (python3 --> python2.7 problem)
    rnn_model = lstm(hidden_units=hidden_units, dense_units=dense_units, 
                    reg=reg, dropout_rate=dropout_rate,
                    seq_length=seq_length, num_classes=num_classes)

    # setup optimizer: ADAM algorithm
    optimizer = Adam(lr=learning_rate, decay=decay)

    # metrics for judging performance of model
    metrics = ['categorical_accuracy'] # ['accuracy']  # if using 'top_k_categorical_accuracy', must specify k

    rnn_model.compile(loss='categorical_crossentropy', optimizer=optimizer,
        metrics=metrics)


    # load saved weights
    # folder_path = 'experiments/base_model/base_model_dropout_rate_0.00e+00/checkpoints/'
    folder_path = 'experiments/base_model/base_model_dropout_rate_3.00e-01/checkpoints/'

    saved_weights = os.path.join(folder_path, 'lstm_weights.0300-0.619.hdf5')
    rnn_model.load_weights(saved_weights)

    # load and prepare test set
    dataset = DataSet(cnn_model)

    X_train, Y_train = dataset.generate_data('train')
    X_val, Y_val = dataset.generate_data('validation')
    X_test, Y_test = dataset.generate_data('test')

    score = rnn_model.evaluate(x=X_train, y=Y_train, verbose=1)
    print("Train Loss: %2.3f" % score[0])
    print("Train Accuracy: %1.3f\n" % score[1])

    score = rnn_model.evaluate(x=X_val, y=Y_val, verbose=1)
    print("Val Loss: %2.3f" % score[0])
    print("Val Accuracy: %1.3f\n" % score[1])

    score = rnn_model.evaluate(x=X_test, y=Y_test, verbose=1)
    print("Test Loss: %2.3f" % score[0])
    print("Test Accuracy: %1.3f\n" % score[1])


    # --- MISCLASSIFICATION ANALYSIS --- #
    Y_pred_class = rnn_model.predict_classes(X_test)
    Y_test_class = np.argmax(Y_test, axis=1)


    target_names = ['backhand', 'bvolley', 'forehand', 'fvolley', 
                   'service', 'smash']
    print classification_report(Y_test_class, Y_pred_class, target_names=target_names)
    conf_matrix = confusion_matrix(Y_test_class, Y_pred_class)

    sns.set(font_scale=1.7)
    df_cm = pd.DataFrame(conf_matrix, index = [i for i in target_names],
                      columns = [i for i in target_names])
    plt.figure(figsize = (8,5))
    ax = sns.heatmap(df_cm, annot=True)
    ax.set_xlabel('Predicted Class', fontsize=18, labelpad=20)
    ax.set_ylabel('True Class', fontsize=18, rotation=0, labelpad=55)
    plt.show()

