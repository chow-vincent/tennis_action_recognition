#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 07:06:59 2018

@author: odibua
"""
import copy
import numpy as np 
import matplotlib.pyplot as plt

def trainingData(maxFrameNum,cellSize,strokeFolder,strokeINDC,stroke,player,strokeNum):
    fileName = './dynamicWordFeaturesCellSize'+str(cellSize)+'/';
    playerID = [];
    strokeID = [];
    nFrames = [];
    X = np.reshape(np.array([''.join(['A0' for j in range(cellSize**2)])]*maxFrameNum),(1,maxFrameNum));
    
    #Store dynamic word data and their associated category in arrays. Also store player and stroke ids.
    Y = np.array([0]);
    cnt = 0;
    for strokeFold in strokeFolder:
        for strokeName in stroke:
            if (np.where(stroke==strokeName)[0] == np.where(strokeFolder==strokeFold)[0]):
                for playerNum in player:
                    for strokeNm in strokeNum:
                        out = np.load(fileName+strokeFold+"/"+"p"+playerNum+"_"+strokeName+"_"+"s"+strokeNm+".npz");
                        playerID.append("p"+playerNum+"_"+strokeName+"_"+"s"+strokeNm);
                        strokeID.append(strokeFold)
                        dynWordList = out['dynWordList'];
                        nFrames.append(len(dynWordList ));
                        if (cnt == 0):
                            X[cnt,(maxFrameNum-len(dynWordList)):] = [dynWord for dynWord in dynWordList]
                        else:
                            X = np.vstack([X,np.array([''.join(['A0' for j in range(cellSize**2)])]*maxFrameNum)]);
                            X[cnt,(maxFrameNum-len(dynWordList)):] = [dynWord for dynWord in dynWordList];
                            Y = np.vstack([Y,strokeINDC[np.where(strokeFold==strokeFolder)[0]]])
                        cnt = cnt+1

    return (X,Y,playerID,strokeID) 

#Obtain dynamic feature data from select folders, players, and strokes.s
def getDynamicFeatureTraining():
    nPlayers=55; nStrokes = 3;
    strokeFolderNames= np.array(['backhand','backhand_slice','backhand_volley','backhand2hands','flat_service',
                    'forehand_flat','forehand_openstands','forehand_slice','forehand_volley','kick_service',
                    'slice_service','smash'])
    strokeNames =  np.array(['backhand','bslice','bvolley','backhand2h','serflat',
                    'foreflat','foreopen','fslice','fvolley','serkick',
                    'serslice','smash'])
    strokeIDX = np.array(range(len(strokeFolderNames)));
    players = np.array(range(nPlayers))+1;
    strokeNums = np.array(range(nStrokes))+1;
    
    
    strokeFolder = copy.deepcopy(strokeFolderNames); 
    stroke = np.array(strokeNames[np.array([idx for idx in range(len(strokeFolderNames)) if strokeFolderNames[idx] in strokeFolder])])
    strokeINDC = np.array(strokeIDX[np.array([idx for idx in range(len(strokeFolderNames)) if strokeFolderNames[idx] in strokeFolder])])
    player = players;#np.array([6]);
    strokeNum = strokeNums;# np.array([2]); 
    cellSize = 7;
    maxFrameNum = 35;
    XTrain,YTrain,playerID,strokeID = trainingData(maxFrameNum,cellSize,strokeFolder,strokeINDC,stroke,player.astype(str),strokeNum.astype(str))                      
    
    return (XTrain,YTrain,playerID,strokeID) 
                        

