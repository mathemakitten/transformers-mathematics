""" Common character-level preprocessing pipeline for all models. """

import numpy as np
import pickle
import glob
import os

from tensorflow.keras.preprocessing.text import Tokenizer

filepaths = 'mathematics_dataset-v1.0/train-medium/*.txt'

# Max input 160 Max output 30

files = glob.glob(filepaths)


def get_universal_encoding(files):

    TOKENIZER_PATH = 'artifacts/tokenizer.pkl'

    if not os.path.isfile(TOKENIZER_PATH):

        list_of_texts = []
        tokenizer = Tokenizer(filters='\n', char_level=True)

        for file in files:
            with open(file, 'r') as f:
                list_of_texts.append(f.read())

        all_text = ''.join(list_of_texts)

        print("Encoding vocabulary")
        tokenizer.fit_on_texts(all_text)

        # Save token dictionary as a json file
        pickle.dump(tokenizer, open(TOKENIZER_PATH, 'wb'))
    else:
        tokenizer = pickle.load(open(TOKENIZER_PATH, 'rb'))

    return tokenizer


tokenizer = get_universal_encoding(files)
'''
import json
token_dict = json.loads(tokenizer.to_json())
len(json.loads(token_dict['config']['word_index']))
'''
questions = []
answers = []

for file in files[0:1]:

    with open(file, 'r') as f:
        lines = f.readlines()

    num_pairs = len(lines) // 2
    '''
    num_pairs = len(lines) // 2
    questions = np.zeroes(size=(num_pairs, 160))
    answers = np.zeroes(size=(num_pairs, 30))
    '''
    for i in range(num_pairs):
        question = lines[i]
        answer = lines[i+1]

        questions.append(question)
        answers.append(answer)

questions_encoded = tokenizer.texts_to_sequences(questions)
answers_encoded = tokenizer.texts_to_sequences(answers)



print('hello')