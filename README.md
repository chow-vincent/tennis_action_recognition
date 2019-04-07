# Action Recognition in Tennis Using Deep Neural Networks

## Authors: Vincent Chow and Ohi Dibua 

This is the code repository for our [CS230: Deep Learning Final Project](http://cs230.stanford.edu/files_winter_2018/projects/6945761.pdf). 

Our goal was to apply deep learning techniques to classify videos of players performing tennis strokes (e.g. forehand, backhand, service).

The [dataset](http://thetis.image.ece.ntua.gr/) consists of 1980 RGB videos sized 640 x 480. In each clip, a player performs one of 12 possible tennis strokes.  

We attempt two different approaches. 

In the first approach, we extract features from video frames using the Inception V3 network pre-trained on ImageNet. These features are then fed into a many-to-many LSTM network, whose softmax outputs are averaged across frames to obtain a final video prediction.

In the second approach, we use standard computer vision techniques. We calculate the optical flow of each video frame, and construct dynamic word representations. These features are then fed into a many-to-one LSTM network, whose final softmax output is used to obtain a video prediction.

Consolidating the data into 6 classes of basic tennis strokes (forehand, backhand, forehand volley, backhand volley, service, and smash), we are able to achieve a performance of 82.3% on test set data using the LRCNN approach.

## LRCNN

The **lrcnn** folder contains code used in the Inception V3 CNN + LSTM approach.

## Optical Flow

The **optical_flow** folder contains code used in the optical flow + LSTM approach.

