#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 21:05:54 2018

@author: odibua
"""

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
import copy

def get_training_dat(trainPerc,validPerc,testPerc): 
    #Load training,validation, and test data
#    trainPerc=0.8;   #Define percent for training
#    validPerc=0.1;
#    testPerc=0.1;
    
    X_train,y_train,playerID,strokeID = getDynamicFeatureTraining();
    y_train_new = copy.deepcopy(y_train);
    y_train_old = copy.deepcopy(y_train);
    
    backhand_idx = np.array([0,1,3]);
    backhand_volley_idx = np.array([2]);
    serve_idx = np.array([4,9,10]);
    forehand_idx = np.array([5,6,7]);
    forehand_volley_idx = np.array([8]);
    smash_idx = np.array([11]);
    
    newClassList = [backhand_idx,backhand_volley_idx,serve_idx,forehand_idx,forehand_volley_idx,smash_idx];
    cntCheck = 0;
    for k in range(len(newClassList)):
        plt.figure();
        new_idx = [old_idx for _,old_class in enumerate(newClassList[k]) for old_idx in range(len(y_train)) if y_train[old_idx]==old_class];
        y_train_new[new_idx] = k;
        cntCheck += len( new_idx);
        #plt.plot(new_idx,y_train_new[new_idx]);
    y_train = y_train_new;
    #Load dynamic words data
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
    
    #Convert training,validation, and test data to 3D matrices.
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
     
    #Normalize the obtained data with the mean and standard deviation of the training set       
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
    
    return (X_train,X_validation,X_test,y_train,y_validation,y_test,playerID,strokeID)