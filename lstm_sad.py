""" The LSTM we wanted but not the LSTM we needed (built with the functional API, bad for inference).
Please see the implementation in lstm.py for what we actually ended up doing. """

from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np

from preprocessing import char2idx, questions_encoded, answers_encoded
from constants import VOCAB_SIZE, QUESTION_MAX_LENGTH, ANSWER_MAX_LENGTH

# Constants
NUM_EPOCHS = 10
BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 100
EMBEDDING_DIM = 32

num_encoder_tokens = VOCAB_SIZE
num_decoder_tokens = VOCAB_SIZE
latent_dim = EMBEDDING_DIM

pad_value = len(char2idx)  # pad with unseen character
questions_pad = [q + [pad_value] * (QUESTION_MAX_LENGTH - len(q)) for q in questions_encoded]
answers_pad = [a + [pad_value] * (ANSWER_MAX_LENGTH - len(a)) for a in answers_encoded]
answers_shifted_pad = np.concatenate([np.expand_dims(np.ones(len(answers_pad))*53, axis=1), np.array(answers_pad)[:, :-1]], axis=1)  # right-shift targets
questions_mask = np.where(np.array(questions_pad) == pad_value, 0, 1)
answers_mask = np.where(np.array(answers_pad) == pad_value, 0, 1)

# Define an input sequence and process it.
encoder_inputs = Input(shape=(None,), name='encoder_inputs')
mask_inputs = Input(shape=(None,), dtype='bool', name='encoder_input_mask')
x = Embedding(input_dim=num_encoder_tokens, output_dim=latent_dim, name='input_embedding')(encoder_inputs)
x, state_h, state_c = LSTM(latent_dim, return_state=True)(x, mask=mask_inputs)
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None,))
mask_labels = Input(shape=(None,), dtype='bool')
x = Embedding(num_decoder_tokens, latent_dim, name='output_embedding')(decoder_inputs)
x = LSTM(latent_dim, return_sequences=True, name='decoder_lstm')(x, initial_state=encoder_states, mask=mask_labels)
decoder_outputs = Dense(num_decoder_tokens, activation='softmax')(x)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, mask_inputs, decoder_inputs, mask_labels], decoder_outputs)

# Compile & run training
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Note that `decoder_target_data` needs to be one-hot encoded,
# rather than sequences of integers like `decoder_input_data`!

model.summary()

model.fit([np.array(questions_pad), np.array(questions_mask),
           np.array(answers_shifted_pad), np.array(answers_mask)],
          np.array(answers_pad),
          batch_size=BATCH_SIZE,
          epochs=NUM_EPOCHS,
          validation_split=0.2)

#model.evaluate()

print('Done!')
