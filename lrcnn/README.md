# Long-term Recurrent Convolutional Neural Network

In this directory you will find everything you need to train and evaluate the LRCNN model on the THETIS dataset.


NOTE: Training and model evaluation code below will not run unless the features
extracted from the Inception V3 network are stored in .npy files in the **data/sequences** directory. 

Due to file storage limits by GitHub, I am unable to push this data. Please contact me if you would like to access these already generated features (~230 MB sized file).

Otherwise, you can generate these files yourself by running the **extract_and_save_sequences.ipynb** file, which will extract features for the samples listed in the **data_file.csv** file. This takes approximately 12 seconds per video, and over 6 hours for the entire dataset.


## Model Training


To train the LRCNN model yourself, you can run the following:
 `python lrcnn_train.py`

 By default, this will load the hyperparameters found in the params.json file in the **experiments/base_model** directory. 

Weights will be saved every 50 epochs in a **checkpoints** folder, and a **train_history.pkl** file containing learning curve data will be saved. 

You can also visualize learning curves during training by entering the following command while in the **lrcnn** directory:
`tensorboard --logdir=experiments`


## Hyperparameter Tuning

To perform a grid search over hyperparameters, you can run the following:
 `python search_hyperparams.py`
 
 You will need to edit the file to adjust what hyperparameters you want to tune and over what ranges.


## Model Evaluation

To evaluate the model you have trained, you can run `python lrcnn_evaluate.py`. This will print (1) the categorical accuracies of the model on the training/validation/test sets, (2) a classification report, and (3) a confusion matrix.

You will have to edit this file to take in the hyperparameters you used, and the name of your weights file. It is currently set to evaluate the best model we were able to achieve in our report.


## data_to_csv.ipynb

This file was used to generate the training/validation/test set splits for the THETIS dataset. The generated **data_file.csv** in the **data** directory contains this information.


## extract_and_save_sequences.ipynb

This file will extract and save features using the Inception V3 CNN network, pretrained on ImageNet. The script will only extract features from samples listed in **data_to_csv.ipynb**.

Sequences are .npy files stored in the **data/sequences** directory within the respective class of tennis stroke.


## data_utils.py

This file contains helper functions for preparing, processing, and manipulating data.


## utils.py

This file is from the CS230 project starter code. Contains useful functions for running hyperparameter tuning experiments.
