""" LSTM Encoder-Decoder architecture with the Keras imperative API.  """
# https://github.com/yusugomori/deeplearning-tf2/blob/master/models/encoder_decoder_lstm.py

import tensorflow as tf
import numpy as np
import os
import argparse
from constants import ANSWER_MAX_LENGTH
from preprocessing import idx2char  # TODO cache questions/answers_encoded as .npy files
from config import *

print(delimiter_token)
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--eager', metavar='eager_mode', type=bool, default=True, help='Eager mode on, else Autograph')
parser.add_argument('--gpu_id', metavar='gpu_id', type=str, default="1", help='The selected GPU to use, default 1')
args = parser.parse_args()

tf.config.experimental_run_functions_eagerly(args.eager)
os.environ['CUDA_VISIBLE_DEVICES'] = "2"

# TODO ADD SLACKBOT, get Ray's code

# load pre-padded data
questions_encoded = np.array(np.load('cache/questions_encoded_padded.npy'))
answers_encoded = np.array(np.load('cache/answers_encoded_padded.npy'))
dataset = tf.data.Dataset.from_tensor_slices((questions_encoded, answers_encoded))
input_data = dataset.take(NUM_EXAMPLES).shuffle(questions_encoded.shape[0]).repeat(NUM_EPOCHS).batch(BATCH_SIZE) \
            .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
NUM_TRAINING_EXAMPLES = int(NUM_EXAMPLES*(1-p_test))
train_data = input_data.take(NUM_TRAINING_EXAMPLES)
valid_data = input_data.skip(NUM_TRAINING_EXAMPLES)

# #  load data
# questions_encoded = np.array(np.load('cache/questions_encoded.npy', allow_pickle=True))
# answers_encoded = np.array(np.load('cache/answers_encoded.npy', allow_pickle=True))
# questions_tensor = tf.ragged.constant(questions_encoded)
# answers_tensor = tf.ragged.constant(answers_encoded)
# dataset = tf.data.Dataset.from_tensor_slices((questions_tensor, answers_tensor))
# input_data = dataset.take(TRAINING_EXAMPLES).shuffle(questions_encoded.shape[0]).repeat(NUM_EPOCHS)\
#              .padded_batch(BATCH_SIZE, padded_shapes=([None,], [None,]))\
#              .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


class Encoder(tf.keras.layers.Layer):

    def __init__(self, input_dim, embedding_size, hidden_dim):
        super(Encoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(input_dim, embedding_size, mask_zero=False)  # can't mask with cuda
        self.lstm = tf.keras.layers.LSTM(hidden_dim, return_state=True)  # return hidden states & sequences

    def call(self, x):
        input_mask = tf.dtypes.cast(tf.clip_by_value(x, 0, 1), dtype=tf.bool)
        input = self.embedding(x)
        output, hidden_state, cell_state = self.lstm(input, mask=input_mask)
        return [hidden_state, cell_state]


class Decoder(tf.keras.layers.Layer):
    def __init__(self, embedding_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(output_dim, embedding_dim, mask_zero=False)
        self.lstm = tf.keras.layers.LSTM(hidden_dim, return_state=True, return_sequences=True)   # apply data at each timestep?
        self.decoder_output = tf.keras.layers.Dense(output_dim)

    def call(self, x, encoder_states):
        # input_mask = tf.dtypes.cast(tf.clip_by_value(tf.expand_dims(x, axis=-1), 0, 1), tf.float32)
        x = self.embedding(x)
        # x = input_mask * x
        x, hidden_state, cell_state = self.lstm(inputs=x, initial_state=encoder_states)
        y = self.decoder_output(x)

        return y, [hidden_state, cell_state]


class EncoderDecoder(tf.keras.Model):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim, max_len):
        super(EncoderDecoder, self).__init__()
        self.encoder = Encoder(input_dim, embedding_dim, hidden_dim)
        self.decoder = Decoder(embedding_dim, hidden_dim, output_dim)

        self.max_length = max_len
        self.output_dim = output_dim

    def call(self, inputs, targets=None):  # targets = None if inference mode

        outputs = tf.zeros((inputs.shape[0], 0, self.output_dim), dtype=tf.float32)
        output_tokens = []
        batch_size = inputs.shape[0]
        len_target_sequences = self.max_length

        decoder_states = self.encoder(inputs)  # initialize decoder lstm states with encoder states

        decoder_output_token = tf.ones((batch_size, 1))  # start token (1) for inference
        for timestep in range(len_target_sequences):
            if targets is not None:  # train
                decoder_output, decoder_states = self.decoder(tf.expand_dims(targets[:, timestep], axis=1), decoder_states)
            else:  # inference
                decoder_output, decoder_states = self.decoder(decoder_output_token, decoder_states)
                decoder_output_token = tf.argmax(decoder_output, axis=-1, output_type=tf.int32)
                output_tokens.append(decoder_output_token)
            outputs = tf.concat([outputs, decoder_output], axis=1)

        return outputs, output_tokens

if __name__ == '__main__':
    np.random.seed(1234)
    tf.random.set_seed(1234)

    model = EncoderDecoder(input_dim=VOCAB_SIZE, embedding_dim=EMBEDDING_SIZE, hidden_dim=LSTM_HIDDEN_SIZE, output_dim=VOCAB_SIZE, max_len=ANSWER_MAX_LENGTH)
    optimizer = tf.keras.optimizers.Adam(lr=0.001)
    tf.keras.utils.Progbar

    @tf.function
    def train_step(inputs, targets, model):
        with tf.GradientTape() as tape:
            # targets[:, :-1] to limit output to 30 chars from 31
            outputs, _ = model(inputs[:, :], targets[:, :-1])  # softmax outputs
            loss_mask = tf.dtypes.cast(tf.clip_by_value(targets[:, 1:], 0, 1), tf.float32)
            # targets[:, 1:] to remove start token so model predicts target's actual chars
            loss = tf.keras.losses.sparse_categorical_crossentropy(targets[:, 1:], outputs, from_logits=True)
            loss = loss * loss_mask
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return tf.reduce_sum(loss)/tf.reduce_sum(loss_mask)

    def inference_step(inputs, model):
        outputs, output_tokens = model(inputs)
        return outputs, output_tokens

    def train(input_data):
        progress_bar = tf.keras.utils.Progbar(int(NUM_TRAINING_EXAMPLES / BATCH_SIZE * NUM_EPOCHS), verbose=1)
        for i, data in enumerate(input_data):
            inputs = data[0]
            targets = data[1]
            progress_bar.update(i)
            loss = train_step(inputs, targets, model)
            if i % 10 == 0:
                print(f' Train loss at batch {i}: {loss}')

    def output_to_tensor(tokens):
        tensor_tokens = tf.squeeze(tf.convert_to_tensor(tokens), axis=2)
        return tf.transpose(tensor_tokens)

    def token_to_text(batch_tensor):
        batch_array = batch_tensor.numpy()
        text_outputs = []
        for sequence_pred in batch_array:
            text = ''.join([idx2char[pred] for pred in sequence_pred])
            text_outputs.append(text)
        return text_outputs

    def inference(input_data):
        for i, data in enumerate(input_data):
            inputs = data[0]
            targets = data[1]
            if i == 0:
                outputs, output_tokens = inference_step(inputs[:, :], model)
                inputs = token_to_text(inputs[:, :])
                targets = token_to_text(targets)
                predictions = token_to_text(output_to_tensor(output_tokens))
                for sample_index in range(len(inputs)):
                    print(f'Input: {inputs[sample_index]}')
                    print(f'Target: {targets[sample_index]}')
                    print(f'Prediction: {predictions[sample_index]} \n')

    train(input_data)
    inference(input_data)