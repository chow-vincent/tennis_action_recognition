import random
##!/usr/bin/env python2
## -*- coding: utf-8 -*-
#"""
#Created on Fri Feb 23 08:05:26 2018
#
#@author: odibua
#"""
import math
import numpy as np
from obtainTrainingData import *
import tensorflow as tf
from tensorflow.contrib import rnn
from getTrainingData import get_training_dat
import copy

######################################################################################################
#Load training,validation, and test data
trainPerc=0.8;   #Define percent for training
validPerc=0.1;
testPerc=0.1;

#Load dynamic words data and player information
X_train,y_train,playerID,strokeID = getDynamicFeatureTraining();
y_train_new = copy.deepcopy(y_train);
y_train_old = copy.deepcopy(y_train);

#Create new consolidated classes
backhand_idx = np.array([0,1,3]);
backhand_volley_idx = np.array([2]);
serve_idx = np.array([4,9,10]);
forehand_idx = np.array([5,6,7]);
forehand_volley_idx = np.array([8]);
smash_idx = np.array([11]);

#Change training labels to consolidated classes
newClassList = [backhand_idx,backhand_volley_idx,serve_idx,forehand_idx,forehand_volley_idx,smash_idx];
cntCheck = 0;
for k in range(len(newClassList)):
    new_idx = [old_idx for _,old_class in enumerate(newClassList[k]) for old_idx in range(len(y_train)) if y_train[old_idx]==old_class];
    y_train_new[new_idx] = k;
    
y_train = y_train_new;

data = np.column_stack([X_train,np.reshape(y_train,(X_train.shape[0],1))]);
nClass = max(y_train)+1; #Define number of classes

#Initialize training empties 
y_trainList = []; 
y_testList = [];
idx_train=[];
idx_validation=[];
idx_test=[];

#Obtain indexes of data for training, validation, and test sets
for j in range(nClass):
    idx=np.where(y_train==j)[0];
    np.random.shuffle(idx);
    idx_train.append(idx[0:int(trainPerc*len(idx))]);
    idx_validation.append(idx[int(trainPerc*len(idx)):len(idx)-int(validPerc*len(idx))]);
    idx_test.append(idx[int((trainPerc+validPerc)*len(idx)):]);
    #idxTest.append(idx[int(trainPerc*len(idx)):]);

#Create arrays for these indices and shuffle them    
idx_train = np.concatenate(idx_train);
idx_validation = np.concatenate(idx_validation);  
idx_test = np.concatenate(idx_test);   
np.random.shuffle(idx_train);
np.random.shuffle(idx_validation);
np.random.shuffle(idx_test);

#Define validation, test, and training data
X_validation = X_train[idx_validation];
X_test = X_train[idx_test];
X_train = X_train[idx_train];

y_validation = y_train[idx_validation];
y_test = y_train[idx_test];
y_train = y_train[idx_train];

#Create 3D matrices that can be sent into an LSTM
X3DTrain = np.zeros((X_train.shape[0],X_train.shape[1],len(X_train[0][0])));
X3DValidation = np.zeros((X_validation.shape[0],X_validation.shape[1],len(X_validation[0][0])));
X3DTest = np.zeros((X_test.shape[0],X_test.shape[1],len(X_test[0][0])));

#Convert training,validation, and test data to 3D matrices with numerical features
temp = np.array([''.join(['A0' for j in range(int(len(X_train[0][0])/2))])]);
for j in range(X3DTrain.shape[0]):
    for k in range(X3DTrain.shape[1]):
        if (X_train[j][k][0:] != temp):
            X3DTrain[j,k,:] = [ord(X_train[j][k][i]) for i in range(len(X_train[0][2]))]
for j in range(X3DValidation.shape[0]):
    for k in range(X3DValidation.shape[1]):
        if (X_validation[j][k][0:] != temp):
            X3DValidation[j,k,:] = [ord(X_validation[j][k][i]) for i in range(len(X_validation[0][2]))]
for j in range(X3DTest.shape[0]):
    for k in range(X3DTest.shape[1]):
        if (X_test[j][k][0:] != temp):
            X3DTest[j,k,:] = [ord(X_test[j][k][i]) for i in range(len(X_test[0][2]))]
            
X3DData = np.concatenate((X3DTrain,X3DValidation,X3DTest),axis=0);
mn3DData = np.mean(X3DData ,axis=0);
std3DData  = np.std(X3DData ,axis=0);

X3DTrain = (X3DTrain - mn3DData)/std3DData ;   
X3DTrain[np.isnan(X3DTrain)]=0;  
X3DValidation = (X3DValidation- mn3DData)/std3DData ;   
X3DValidation[np.isnan(X3DValidation)]=0;     
X3DTest = (X3DTest - mn3DData)/std3DData ;   
X3DTest[np.isnan(X3DTest)]=0;    

#Define training, test and validation data that will be used in Deep Learning
idx_first = 0; #Define number of sequences to use   
nFeatures = X3DTrain.shape[-1]; #Number of features 
X_train = X3DTrain[:,-idx_first:,:];
X_validation = X3DValidation[:,-idx_first:,:];
X_test = X3DTest[:,-idx_first:,:]; 
y_train = y_train.astype(int)
y_validation = y_validation.astype(int)
y_test = y_test.astype(int)

#Count number of sets in each data partition
training_data_count = len(X_train)  
validation_data_count = len(X_train)  
test_data_count = len(X_test) 
######################################################################################################
#Define parameters relevant to the data
n_steps = X_train.shape[1];#len(X_train[0])  # 128 timesteps per series
n_input = X_train.shape[2];#len(X_train[0][0])  # 9 input parameters per timestep
n_classes = len(np.unique(y_train)); # Total classes (should go up, or should go down)

#Define hyperparameters of layers
n_hidden =  128;
n_layers=2;
forget_bias_in=1.0;

# Optimization parameters
beta1=0.9,
beta2=0.999,
epsilon=1e-08,
learning_rate = 1e-3;#0.0025 #learning rate in AdamOptimizer

#Regularization Parameters
lambda_loss_l2_amount = 0.015; #L2 lambda loss
lambda_loss_l1_amount = 0.0015; #L1 lambda loss
dropout_prob = 0.7;

#Define batch size
batch_size = 128;
n_epoches = 300;
#Define number of training iterations
training_iters = (training_data_count/batch_size)*n_epoches;# * 200#100  # Loop 300 times on the dataset 
######################################################################################################
#Define cases that modify implementation of LSTM
#Regularization options
l2_idx=0;
l1_idx=1;
dropout_idx=2;
none_idx=3;
regularization_choice = dropout_idx;#dropout_idx;#l2_idx;

#Definte type of LSTM
many_to_one_idx = 0;
many_to_many_idx = 1;
lstm_type_choice = many_to_one_idx;#many_to_one_idx;
if (lstm_type_choice==many_to_many_idx):
    y_train=np.repeat(y_train,X_train.shape[1],axis=1);
    y_test=np.repeat(y_test,X_train.shape[1],axis=1);
    y_validation=np.repeat(y_validation,X_train.shape[1],axis=1);
######################################################################################################
display_iter = 300  # To show test set accuracy during training
#Define LSTM_RNN function that produces outputs from LSTM
def LSTM_RNN(_X, _weights, _biases, n_layers,forget_bias_in,lstm_type_choice):
    # Function returns a tensorflow LSTM (RNN) artificial neural network from given parameters. 
    # Moreover, two LSTM cells are stacked which adds deepness to the neural network. 
    # Note, some code of this notebook is inspired from an slightly different 
    # RNN architecture used on another dataset, some of the credits goes to 
    # "aymericdamien" under the MIT license.

    # (NOTE: This step could be greatly optimised by shaping the dataset once
    # input shape: (batch_size, n_steps, n_input)
    _X = tf.transpose(_X, [1, 0, 2])  # permute n_steps and batch_size
    # Reshape to prepare input to hidden activation
    _X = tf.reshape(_X, [-1, n_input]) 
    # new shape: (n_steps*batch_size, n_input)
    
    # Linear activation
    _X = tf.nn.relu(tf.matmul(_X, _weights['hidden']) + _biases['hidden'])
    # Split data because rnn cell needs a list of inputs for the RNN inner loop
    _X = tf.split(_X, n_steps, 0) 
    # new shape: n_steps * (batch_size, n_hidden)

    #Define k stacked LSTM cells with tensor flow 
    cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=forget_bias_in, state_is_tuple=True)
    if (regularization_choice == dropout_idx):
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=dropout_prob)
    lstm_cells = tf.contrib.rnn.MultiRNNCell([cell for _ in range(n_layers)], state_is_tuple=True)
    
    
    #outputs, states = tf.contrib.rnn.static_rnn(lstm_cells, _X, dtype=tf.float32);
    # Get LSTM cell output
    if (lstm_type_choice==many_to_one_idx):
        # Get last time step's output feature for a "many to one" style classifier, 
        # as in the image describing RNNs at the top of this page
        outputs, states = tf.contrib.rnn.static_rnn(lstm_cells, _X, dtype=tf.float32);
        lstm_last_output = outputs[-1];
        return tf.matmul(lstm_last_output, _weights['out']) + _biases['out']
              
    elif (lstm_type_choice==many_to_many_idx):     
        # Get last time step's output feature for a "many to one" style classifier, 
        # as in the image describing RNNs at the top of this page
        #lstm_last_output = outputs;
        outputs, states = tf.contrib.rnn.static_rnn(lstm_cells, _X, dtype=tf.float32);
        outputs = tf.reshape(outputs, [-1, n_hidden])
        return tf.matmul(outputs, _weights['out']) + _biases['out']
######################################################################################################    
#Supporting functions    
#Define function that extracts batches from training data
def extract_batch_size(_train, step, batch_size):
    # Function to fetch a "batch_size" amount of data from "(X|y)_train" data.   
    shape = list(_train.shape)
    shape[0] = batch_size
    batch_s = np.empty(shape)

    for i in range(batch_size):
        # Loop index
        index = ((step-1)*batch_size + i) % len(_train)
        batch_s[i] = _train[index] 

    return batch_s

#One Hot implementation with lstm type defined
def one_hot(y_,lstm_type_choice):
    # Function to encode output labels from number indexes 
    # e.g.: [[5], [0], [3]] --> [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]
    if (lstm_type_choice==many_to_one_idx):
        y_ = y_.reshape(len(y_))
        
        n_values = int(np.max(y_train)) + 1 #States that 
        return np.eye(n_values)[np.array(y_, dtype=np.int32)]  # Returns FLOATS
    elif (lstm_type_choice == many_to_many_idx):
        return y_
######################################################################################################
#Initialize weights and placeholder variables
# Graph input/output placeholders
x = tf.placeholder(tf.float32, [None, n_steps, n_input])
if (lstm_type_choice==many_to_many_idx):
    y = tf.placeholder(tf.int32, [None,n_steps, n_classes]);
    y_val = tf.placeholder(tf.int32, [None,n_steps, n_classes]);
else:
    y = tf.placeholder(tf.int32, [None, n_classes]);#tf.placeholder(tf.float32, [None, n_classes])

#Initialize with special version of Xavier He
limit_hidden = np.sqrt(6.0/(n_input+n_hidden)); 
limit_out = np.sqrt(6.0/(n_classes+n_hidden)); 
weights = {
    'hidden': tf.Variable(tf.random_uniform([n_input, n_hidden],minval=-limit_hidden,maxval=limit_hidden)), # Hidden layer weights
    'out': tf.Variable(tf.random_uniform([n_hidden, n_classes],minval=-limit_out,maxval=limit_out))
}
biases = {
    'hidden': tf.zeros([n_hidden]),
    'out': tf.zeros([n_classes])
}
######################################################################################################

#Do softmax of predictions
pred = tf.convert_to_tensor(LSTM_RNN(x, weights, biases,n_layers,forget_bias_in,lstm_type_choice))


#Choose regularization and regularization cost
if (regularization_choice==l2_idx):
    l2 = lambda_loss_l1_amount * sum(
        tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables()
    ) # L2 loss prevents this overkill neural network to overfit the data
    regularizer_cost=l2;
elif (regularization_choice==l1_idx):
    l1 = tf.contrib.layers.l1_regularizer(
       scale=lambda_loss_l1_amount, scope=None
    )
   #weights = tf.trainable_variables() # all vars of your graph
    regularization_penalty = tf.contrib.layers.apply_regularization(l1, tf.trainable_variables())
    regularizer_cost=regularization_penalty;#l1;
else:
    regularizer_cost = 0;

#Calculate loss, cost and accuracy. Define optimizaer
print("first pred",pred)
if (lstm_type_choice==many_to_many_idx):
    pred = tf.reshape(pred,[batch_size, n_steps,n_classes])
    y = tf.reshape(y,[batch_size, n_steps])

    loss = tf.contrib.seq2seq.sequence_loss(pred,y,tf.ones([batch_size, n_steps], dtype=tf.float32),average_across_timesteps=True,average_across_batch=True)
    cost = tf.reduce_mean(loss) + regularizer_cost;
    
    softmax_out = tf.nn.softmax(tf.reshape(pred, [-1, n_classes]))
    predict = tf.cast(tf.argmax(softmax_out, axis=1), tf.int32)
    correct_pred = tf.equal(predict, tf.reshape(y, [-1]))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32)) 
elif (lstm_type_choice==many_to_one_idx):
    softMx = tf.nn.softmax(logits=pred);
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred)) + regularizer_cost; # Softmax loss   
    correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32));

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) # Adam Optimizer
######################################################################################################

# To keep track of training's performance
test_losses = []
test_accuracies = []
validation_losses = []
validation_accuracies = []
train_losses = []
train_accuracies = []

# Launch the graph
sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=False));
init = tf.global_variables_initializer()
sess.run(init)

# Perform Training steps with "batch_size" amount of example data at each loop
step = 1
#while step * batch_size <= training_iters:
while step <= training_iters:
    batch_xs =         extract_batch_size(X_train, step, batch_size)
    batch_ys = one_hot(extract_batch_size(y_train, step, batch_size),lstm_type_choice)

    _, loss, acc,corr_pred,predict = sess.run(
        [optimizer, cost, accuracy,correct_pred,pred],
        feed_dict={
            x: batch_xs, 
            y: batch_ys
        }
    )
    train_losses.append(loss)
    train_accuracies.append(acc)
    
    validation_losses.append(loss)
    validation_accuracies.append(acc)

        # To not spam console, show training accuracy/loss in this "if"
    print("Training iter #" + str(step*batch_size) + \
          ":   Batch Loss = " + "{:.6f}".format(loss) + \
          ", Accuracy = {}".format(acc))
    
    # Evaluation on the validation set (no learning made here - just evaluation for diagnosis)
    loss, acc,corr_val = sess.run(
        [cost, accuracy,correct_pred],  
        feed_dict={
            x: X_validation,
            y: one_hot(y_validation,lstm_type_choice) 
        }
    )
    validation_losses.append(loss)
    validation_accuracies.append(acc)
    print("PERFORMANCE ON VALIDATION SET: " + \
          "Batch Loss = {}".format(loss) + \
          ", Accuracy = {}".format(acc))
    
    
    # Evaluation on the test set (no learning made here - just evaluation for diagnosis)
    loss, acc,corr_test = sess.run(
        [cost, accuracy,correct_pred], 
        feed_dict={
            x: X_test,
            y: one_hot(y_test,lstm_type_choice) 
        }
    )
    test_losses.append(loss)
    test_accuracies.append(acc)
   # print(corr_test,np.where(corr_test==False))

    step += 1
# Evaluation on the test set (no learning made here - just evaluation for diagnosis)
loss, acc = sess.run(
    [cost, accuracy], 
    feed_dict={
        x: X_test,
        y: one_hot(y_test,lstm_type_choice) 
    }
)
false_val_idx = idx_validation[np.where(corr_val==False)[0]];
false_test_idx = idx_test[np.where(corr_test==False)[0]];
test_accuracies.append(acc)
np.savez("manyToOne_DropOutVincent2LayerCompressedAcc.npz",false_val_idx=false_val_idx,false_test_idx=false_test_idx,idx_train=idx_train,idx_test=idx_test,idx_validation=idx_validation,y_train=y_train,y_train_old=y_train_old,y_validation=y_validation,y_test=y_test,train_losses=train_losses,train_accuracies=train_accuracies
         ,test_losses=test_losses,test_accuracies=test_accuracies,validation_losses=validation_losses,validation_accuracies=validation_accuracies,n_epoches=n_epoches,batch_size=batch_size,training_iters=training_iters )
print("Optimization Finished!") 
# 
