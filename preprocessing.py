""" Common character-level preprocessing pipeline for all models. """

import tensorflow as tf
import numpy as np
import pickle
import glob
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# question_dir = 'train-easy'
# question_filename = 'arithmetic__add_or_sub.txt'
# filepaths = os.path.join('/media/biggie1/transformers-mathematics/mathematics_dataset-v1.0', question_dir, question_filename)
filepaths = os.path.join('/media/biggie1/transformers-mathematics/mathematics_dataset-v1.0', 'copy_task.txt')
#filepaths = 'mathematics_dataset-v1.0/train-medium/*.txt'

if not os.path.isdir('artifacts'):
    os.mkdir('artifacts')
if not os.path.isdir('cache'):
    os.mkdir('cache')

# Max input 160, Max output 30
files = glob.glob(filepaths)


def get_universal_encoding(files):

    TOKENIZER_PATH = 'artifacts/char2idx.pkl'
    UNTOKENIZER_PATH = 'artifacts/idx2char.pkl'

    if not os.path.isfile(TOKENIZER_PATH):

        list_of_texts = []
        for file in files:
            with open(file, 'r') as f:
                list_of_texts.append(f.read())

        all_text = ''.join(list_of_texts)

        print("Encoding vocabulary")

        # Get the unique characters in the file (vocab)
        chars_to_remove = {'\n'}
        vocab = list(set(all_text) - chars_to_remove)
        # pad, start, stop
        vocab = ['_', '@', '~'] + vocab
        # Creating a mapping from unique characters to indices
        char2idx = {u: i for i, u in enumerate(vocab)}
        idx2char = {v: k for k, v in char2idx.items()}

        # Save token dictionary as a json file
        pickle.dump(char2idx, open(TOKENIZER_PATH, 'wb'))
        pickle.dump(idx2char, open(UNTOKENIZER_PATH, 'wb'))
    else:
        char2idx = pickle.load(open(TOKENIZER_PATH, 'rb'))
        idx2char = pickle.load(open(UNTOKENIZER_PATH, 'rb'))

    return char2idx, idx2char


char2idx, idx2char = get_universal_encoding(files)

if __name__ == "__main__":
    questions_encoded = []
    answers_encoded = []

    for file in files:

        with open(file, 'r') as f:
            lines = f.readlines()

        num_pairs = len(lines) // 2

        for i in range(0, 2 * num_pairs, 2):
            question = lines[i].strip()
            answer = lines[i+1].strip()

            questions_encoded.append([char2idx[q] for q in question])
            answers_encoded.append([char2idx[a] for a in answer])

    # padded
    questions_pad = [q + [0] * (160 - len(q)) for q in questions_encoded]
    # TODO: Clean this code
    # 1: start, 2: stop, 0: pad
    answers_pad = [[1] + a + [2] + [0] * (29 - len(a)) for a in answers_encoded]  # decoder input
    np.save('cache/questions_encoded_padded.npy', np.array(questions_pad))
    np.save('cache/answers_encoded_padded.npy', np.array(answers_pad))

    # # no padding
    # np.save('cache/questions_encoded.npy', np.array(questions_encoded))
    # np.save('cache/answers_encoded.npy', np.array(answers_encoded))

    print('debug')
