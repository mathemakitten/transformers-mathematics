""" Common character-level preprocessing pipeline for all models. """

import tensorflow as tf
import numpy as np
import pickle
import glob
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

filepaths = '/media/biggie1/hn/transformers-mathematics/mathematics_dataset-v1.0/train-medium/tiny_data_1000.txt'
# filepaths = 'mathematics_dataset-v1.0/train-medium/*.txt'

# Max input 160 Max output 30

files = glob.glob(filepaths)


def get_universal_encoding(files):

    TOKENIZER_PATH = 'artifacts/char2idx.pkl'
    UNTOKENIZER_PATH = 'artifacts/idx2char.pkl'

    if not os.path.isfile(TOKENIZER_PATH):

        list_of_texts = []
        #tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='\n', char_level=True)

        for file in files:
            with open(file, 'r') as f:
                list_of_texts.append(f.read())

        all_text = ''.join(list_of_texts)

        print("Encoding vocabulary")
        #tokenizer.fit_on_texts(set(all_text))

        # Get the unique characters in the file (vocab)
        chars_to_remove = {'\n'}
        vocab = list(set(all_text) - chars_to_remove)

        # Creating a mapping from unique characters to indices
        char2idx = {u: i for i, u in enumerate(vocab)}

        idx2char = np.array(vocab)

        # Convert text to indices
        #text_as_int = np.array([char2idx[c] for c in all_text])

        # Save token dictionary as a json file
        pickle.dump(char2idx, open(TOKENIZER_PATH, 'wb'))
        pickle.dump(idx2char, open(UNTOKENIZER_PATH, 'wb'))
    else:
        char2idx = pickle.load(open(TOKENIZER_PATH, 'rb'))
        idx2char = pickle.load(open(UNTOKENIZER_PATH, 'rb'))

    return char2idx, idx2char


char2idx, idx2char = get_universal_encoding(files)
'''
import json
token_dict = json.loads(tokenizer.to_json())
len(json.loads(token_dict['config']['word_index']))
'''
questions_encoded = []
answers_encoded = []

for file in files[0:1]:

    with open(file, 'r') as f:
        lines = f.readlines()

    num_pairs = len(lines) // 2

    for i in range(0, 2 * num_pairs, 2):
        question = lines[i].strip()
        answer = lines[i+1].strip()

        questions_encoded.append([char2idx[q] for q in question])
        answers_encoded.append([char2idx[a] for a in answer])

question_max_length = 160
answer_max_length = 30

pad_value = len(char2idx)  # pad with unseen character
questions_pad = [q + [pad_value] * (question_max_length - len(q)) for q in questions_encoded]
answers_pad = [a + [pad_value] * (answer_max_length - len(a)) for a in answers_encoded]
answers_shifted_pad = np.concatenate([np.expand_dims(np.ones(len(answers_pad))*53, axis=1), np.array(answers_pad)[:, :-1]], axis=1)  # right-shift targets
questions_mask = np.where(np.array(questions_pad) == pad_value, 0, 1)
answers_mask = np.where(np.array(answers_pad) == pad_value, 0, 1)

train_dataset = tf.data.Dataset.from_tensor_slices((questions_pad, answers_pad, answers_shifted_pad))

#questions_encoded = np.array(questions_encoded)
#answers_encoded = np.array(answers_encoded)

# TODO see what these lists looks like, convert to tf.data w/ from_tensor_slices

NUM_EPOCHS = 10
BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 100
EMBEDDING_DIM = 32
VOCAB_SIZE = len(char2idx)+1

train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)#.repeat(NUM_EPOCHS)

# class mlp(tf.keras.layers.Layer):
#
#     def __init__(self, neurons):
#         super(mlp, self).__init__()
#         self.neurons = int(neurons)
#         self.ffn_dim = params.embedding_dim  # still all the same dims...
#         self.hidden1 = tf.keras.layers.Dense(units=self.neurons, activation='relu', name='mlp_h1')
#         self.hidden2 = tf.keras.layers.Dense(units=self.ffn_dim, name='mlp_h2')
#
#     def call(self, x):
#         h = self.hidden1(x)
#         h2 = self.hidden2(h)
#         return h2


#class LSTM(tf.keras.Model)



from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
from tensorflow.keras.models import Model

num_encoder_tokens = VOCAB_SIZE
num_decoder_tokens = VOCAB_SIZE
latent_dim = EMBEDDING_DIM

# Define an input sequence and process it.
encoder_inputs = Input(shape=(None,), name='encoder_inputs')
mask_inputs = Input(shape=(None,), dtype='bool', name='encoder_input_mask')
x = Embedding(input_dim=num_encoder_tokens, output_dim=latent_dim, name='input_embedding')(encoder_inputs)
x, state_h, state_c = LSTM(latent_dim,
                           return_state=True)(x, mask=mask_inputs)
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

#model.fit(train_dataset)

print('hello')
''