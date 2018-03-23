"""
Perform hyperparameter search experiments. 
File adapted from CS230 starter code.
"""

import argparse
import os
from subprocess import check_call
import sys

from utils import Params

import numpy as np
import time


PYTHON = sys.executable
parser = argparse.ArgumentParser()
parser.add_argument('--parent_dir', default='experiments/base_model',
                    help="Directory containing params.json")
# parser.add_argument('--data_dir', default='data/small',
#                     help="Directory containing the dataset")


def launch_training_job(parent_dir, job_name, params):
    """Launch training of the model with a set of hyperparameters in parent_dir/job_name

    Args:
        parent_dir: (string) directory containing config, weights and log
        params: (dict) containing hyperparameters
    """

    # Create a new folder in parent_dir with unique_name "job_name"
    model_dir = os.path.join(parent_dir, job_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Write parameters in json file
    json_path = os.path.join(model_dir, 'params.json')
    params.save(json_path)

    # Launch training with this config
    cmd = "{python} lrcnn_train.py --model_dir {model_dir}".format(python=PYTHON,
            model_dir=model_dir)
    print(cmd)
    check_call(cmd, shell=True)


if __name__ == "__main__":

    start_time = time.time()
    # Load the "reference" parameters from parent_dir json file
    args = parser.parse_args()
    json_path = os.path.join(args.parent_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)


    # --- LEARNING RATES --- #
    # range: [1e-3, 1e-7]
    # learning_rates = [1e-7, 1e-5, 1e-3]  # for now, manually deciding these
    # learning_rates = [10**(-4*np.random.rand()-3) for _ in range(3)]
    # learning_rates = sorted(learning_rates)

    # --- BATCH SIZES --- #
    # range: [16, 1024], and enforce all unique
    # batch_sizes = [2**4, 2**7, 2**10]  # for now, manually deciding these

    # batch_sizes = []
    # while len(batch_sizes) != 3:
    #     rand_int = 2**(np.random.randint(low=4, high=10))
    #     if rand_int not in batch_sizes:
    #         batch_sizes.append(rand_int)
    # # batch_size = [2**(np.random.randint(low=4, high=10)) for _ in range(3)]
    # batch_sizes = sorted(batch_sizes)

    # --- DROPOUT RATE --- #
    dropout_rates = [0.3]

    # --- HIDDEN UNITS --- #
    # num_hidden = [16, 512, 1024]
    # num_hidden = [128, 256]

    # --- DENSE UNITS --- #
    # num_dense = [64, 128, 256]


    # ------ LEARNING RATE & BATCH SIZE EXPERIMENT ------ #
    # for learning_rate in learning_rates:
    #     params.learning_rate = learning_rate  # modify relevant param

    #     for batch_size in batch_sizes:
    #         params.batch_size = batch_size

            # # launch unique job name
            # job_name = "lr_{:1.2e}_bs_{:04d}".format(learning_rate, batch_size)
            # launch_training_job(args.parent_dir, job_name, params)


    # ------ DROPOUT RATE, HIDDEN UNITS, DENSE UNITS EXPERIMENT ------ #
    # for dropout_rate in dropout_rates:
    #     params.dropout_rate = dropout_rate

    #     for hidden_units in num_hidden:
    #         params.hidden_units = hidden_units

    #         for dense_units in num_dense:
    #             params.dense_units = dense_units
    #             job_name = "drop_{:1.2f}_hidden_{:04d}_dense_{:04d}".format(dropout_rate, 
    #                                                                     hidden_units, 
    #                                                                     dense_units)
    #             launch_training_job(args.parent_dir, job_name, params)


    # ------ BASE MODEL REGULARIZED/NON-REGULARIZED ------ #
    for dropout_rate in dropout_rates:
        params.dropout_rate = dropout_rate
        job_name = "base_model_dropout_rate_{:1.2e}".format(dropout_rate)
        launch_training_job(args.parent_dir, job_name, params)


    end_time = time.time()

    # print training time in minutes
    print("Training Time: %8.2f" % ((end_time - start_time)/60.))












