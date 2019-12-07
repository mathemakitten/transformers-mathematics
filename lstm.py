""" LSTM Encoder-Decoder architecture with the Keras imperative API.  """

import tensorflow as tf
import numpy as np
import os

from preprocessing import idx2char  # TODO cache questions/answers_encoded as .npy files
from constants import VOCAB_SIZE, QUESTION_MAX_LENGTH, ANSWER_MAX_LENGTH

tf.config.experimental_run_functions_eagerly(True)

# Hparams — make this less shitty and not here
BATCH_SIZE = 32
LSTM_HIDDEN_SIZE = 128
delimiter_token = VOCAB_SIZE - 1  # == newline_character; which is out-of-vocabulary. vocab starts at 0, so do not +1
NUM_EPOCHS = 1
TRAINING_STEPS = 10000

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# TODO ADD SLACKBOT, get Ray's code

# Much of this code was adapted from
# https://github.com/yusugomori/deeplearning-tf2/blob/master/models/encoder_decoder_lstm.py

# Load input data
questions_encoded = np.array(np.load('cache/questions_encoded_padded.npy'))
answers_encoded = np.array(np.load('cache/answers_encoded_padded.npy'))

#  Input pipeline  # TODO pad in tf.data, not in numpy
dataset = tf.data.Dataset.from_tensor_slices((questions_encoded, answers_encoded))

# input_data = dataset.take(1000).shuffle(questions_encoded.shape[0]).repeat(NUM_EPOCHS).batch(32) #.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

input_data = dataset.take(TRAINING_STEPS).shuffle(questions_encoded.shape[0])\
    .repeat(NUM_EPOCHS)\
    .batch(BATCH_SIZE)
    #\
     #.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

#input_data = input_data.batch(32)

# # Load input data
# questions_encoded = np.array(np.load('cache/questions_encoded.npy', allow_pickle=True))
# questions_encoded = tf.ragged.constant(questions_encoded)
# answers_encoded = np.array(np.load('cache/answers_encoded.npy', allow_pickle=True))
# answers_encoded = tf.ragged.constant(answers_encoded)
#
# #  Input pipeline
# dataset = tf.data.Dataset.from_tensor_slices((questions_encoded, answers_encoded))
#
# input_data = dataset.take(50000).shuffle(questions_encoded.shape[0])\
#     .padded_batch(batch_size=BATCH_SIZE,
#                   padded_shapes=([None], [None]))\
#     .repeat(NUM_EPOCHS).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


class Encoder(tf.keras.layers.Layer):

    def __init__(self, input_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(input_dim, hidden_dim, mask_zero=True)
        self.lstm = tf.keras.layers.LSTM(hidden_dim, return_state=True)  # return hidden states & sequences

    def call(self, x):
        input = self.embedding(x)
        output, hidden_state, cell_state = self.lstm(input)
        return [hidden_state, cell_state]


class Decoder(tf.keras.layers.Layer):
    def __init__(self, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(output_dim, hidden_dim, mask_zero=True)
        self.lstm = tf.keras.layers.LSTM(hidden_dim, return_state=True, return_sequences=True)   # apply data at each timestep?
        self.decoder_output = tf.keras.layers.Dense(output_dim)

    def call(self, x, encoder_states):
        x = self.embedding(x)
        x, hidden_state, cell_state = self.lstm(inputs=x, initial_state=encoder_states)
        y = tf.nn.softmax(self.decoder_output(x))

        return y, [hidden_state, cell_state]


class EncoderDecoder(tf.keras.Model):
    def __init__(self, input_dim, hidden_dim, output_dim, max_len):
        super(EncoderDecoder, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim)
        self.decoder = Decoder(hidden_dim, output_dim)

        self.max_length = max_len
        self.output_dim = output_dim

    def call(self, inputs, targets=None):  # targets = None if inference mode

        outputs = tf.zeros((inputs.shape[0], 0, self.output_dim), dtype=tf.float32)
        output_tokens = []
        batch_size = inputs.shape[0]
        len_target_sequences = self.max_length

        encoder_hidden_state = self.encoder(inputs)
        decoder_hidden_state = encoder_hidden_state

        decoder_output_token = tf.ones((batch_size, 1)) * delimiter_token  # start token for inference
        for timestep in range(len_target_sequences):
            if targets is not None:  # train
                decoder_output, decoder_hidden_state = self.decoder(tf.expand_dims(targets[:, timestep], axis=1), decoder_hidden_state)
            else:  # inference
                decoder_output, decoder_hidden_state = self.decoder(decoder_output_token, decoder_hidden_state)
                decoder_output_token = tf.argmax(decoder_output, axis=-1, output_type=tf.int32)
                output_tokens.append(decoder_output_token)
            outputs = tf.concat([outputs, decoder_output], axis=1)

        return outputs, output_tokens

if __name__ == '__main__':
    np.random.seed(1234)
    tf.random.set_seed(1234)

    model = EncoderDecoder(input_dim=VOCAB_SIZE, hidden_dim=LSTM_HIDDEN_SIZE, output_dim=VOCAB_SIZE, max_len=ANSWER_MAX_LENGTH)
    optimizer = tf.keras.optimizers.Adam()
    tf.keras.utils.Progbar

    @tf.function
    def train_step(inputs, targets, model):
        with tf.GradientTape() as tape:
            outputs, _ = model(inputs, targets)  # softmax outputs
            loss = tf.keras.losses.sparse_categorical_crossentropy(targets, outputs)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

    def inference_step(inputs, model):
        outputs, output_tokens = model(inputs)
        return outputs, output_tokens

    def train(input_data):
        progress_bar = tf.keras.utils.Progbar(int(TRAINING_STEPS / BATCH_SIZE * NUM_EPOCHS), verbose=1)
        for i, data in enumerate(input_data):
            inputs = data[0]
            targets = data[1]
            progress_bar.update(i)
            train_step(inputs, targets, model)

    def output_to_tensor(tokens):
        tensor_tokens = tf.squeeze(tf.convert_to_tensor(tokens), axis=2)
        return tf.transpose(tensor_tokens)

    def token_to_text(batch_tensor):
        batch_array = batch_tensor.numpy()
        text_outputs = []
        for sequence_pred in batch_array:
            text = ''.join([idx2char[pred] if pred != 0 and pred != 44 else '' for pred in sequence_pred])
            text_outputs.append(text)
        return text_outputs

    def inference(input_data):
        for i, data in enumerate(input_data):
            inputs = data[0]
            targets = data[1]
            if i == 0:
                outputs, output_tokens = inference_step(inputs, model)
                inputs = token_to_text(inputs)
                targets = token_to_text(targets)
                predictions = token_to_text(output_to_tensor(output_tokens))
                for sample_index in range(len(inputs)):
                    print(f'Input: {inputs[sample_index]}')
                    print(f'Target: {targets[sample_index]}')
                    print(f'Prediction: {predictions[sample_index]} \n')

    train(input_data)
    inference(input_data)