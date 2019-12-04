""" Common character-level preprocessing pipeline for all models. """

import tensorflow as tf
import numpy as np
import pickle
import glob
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

filepaths = '/media/biggie1/hn/transformers-mathematics/mathematics_dataset-v1.0/train-medium/tiny_data_1000.txt'
# filepaths = 'mathematics_dataset-v1.0/train-medium/*.txt'

# Max input 160, Max output 30
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

