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
# from lstm import inference_step, inference, get_validation_metrics, get_accuracy
from transformer import Transformer

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


if __name__ == '__main__':
    np.random.seed(1234)
    tf.random.set_seed(1234)

    tf.keras.utils.Progbar

    params = TransformerParams()

    logger = get_logger('validation', params.experiment_dir)
    logger.info("Logging to {}".format(params.experiment_dir))

    # preprocess data
    dataset = tf.data.Dataset.from_tensor_slices((questions_encoded, answers_encoded))
    input_data = dataset.take(params.num_examples).shuffle(questions_encoded.shape[0]).batch(params.batch_size) \
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    train_data = input_data.take(params.num_training_batches).repeat(params.num_epochs)
    valid_data = input_data.skip(params.num_training_batches)

    model = Transformer(params)
    model.train(params, train_data, valid_data, logger)
    # model.inference()


'''
HN NOTE: 
For generalizability of training pipeline, 
Train steps should be methods of the model 
and individual train steps should output masked preds + targets 
But the training loop should be general 
Training loop should be similar to lstm.py current one, contain 
- Tensorboard logging 
- Validation loss + accuracy if i % n 
- Early stopping check 
- Outputting samples 
- Model checkpointing 
'''
