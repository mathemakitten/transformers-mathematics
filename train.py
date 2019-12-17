""" Train any model with the same preprocessing, logging, and evaluation infrastructure """

import tensorflow as tf
import numpy as np
import os
import argparse
from constants import ANSWER_MAX_LENGTH
from preprocessing import idx2char  # TODO cache questions/answers_encoded as .npy files
from config import *
from utils import get_logger
import time
from constants import QUESTION_MAX_LENGTH
from lstm import inference_step, inference, get_validation_metrics, get_accuracy

logger = get_logger('validation_log')
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--eager', metavar='eager_mode', type=bool, default=True, help='Eager mode on, else Autograph')
parser.add_argument('--gpu_id', metavar='gpu_id', type=str, default="1", help='The selected GPU to use, default 1')
args = parser.parse_args()

tf.config.experimental_run_functions_eagerly(args.eager)
os.environ['CUDA_VISIBLE_DEVICES'] = "3"

# load pre-padded data
dataset_id = '_all_data_ever'
questions_encoded = np.array(np.load('cache/questions_encoded_padded_{}.npy'.format(dataset_id)))
answers_encoded = np.array(np.load('cache/answers_encoded_padded_{}.npy'.format(dataset_id)))

params = TransformerParams()

dataset = tf.data.Dataset.from_tensor_slices((questions_encoded, answers_encoded))
input_data = dataset.take(params.num_examples).shuffle(questions_encoded.shape[0]).batch(params.batch_size) \
            .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
train_data = input_data.take(params.num_training_batches).repeat(params.num_epochs)
valid_data = input_data.skip(params.num_training_batches)


if __name__ == '__main__':  # TODO HN move these function definitions out of main... hahahahaha yikes
    np.random.seed(1234)
    tf.random.set_seed(1234)
    tf.keras.utils.Progbar
    logger.info("Logging to {}".format(params.experiment_dir))

    model = Model()
    model.train()
    model.inference()