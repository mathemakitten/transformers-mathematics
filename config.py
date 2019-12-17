import os
from constants import VOCAB_SIZE
from datetime import datetime as dt

current_time = dt.now().strftime('%Y%m%d_%H_%M-')

BATCH_SIZE = 256
EMBEDDING_SIZE = 512
LSTM_HIDDEN_SIZE = 512
NUM_EPOCHS = 10
NUM_EXAMPLES = 666666*3
p_test = .2
EXPERIMENT_DIR = os.path.join('experiment_results', current_time + 'transformer_testing')
print(EXPERIMENT_DIR)


class LSTMParams:
    def __init__(self):
        self.batch_size = 256
        self.embedding_size = 512
        self.lstm_hidden_size = 512
        self.num_epochs = 10
        self.num_examples = 666666*3
        self.p_test = 0.2
        self.experiment_dir = os.path.join('experiment_results', current_time + 'lstm_run')
        self.num_training_batches = int(self.num_examples/self.batch_size*(1-self.p_test))

        '''
        BATCH_SIZE = 256
        EMBEDDING_SIZE = 512
        LSTM_HIDDEN_SIZE = 512
        NUM_EPOCHS = 10
        NUM_EXAMPLES = 666666*3
        p_test = .2
        EXPERIMENT_DIR = os.path.join('experiment_results', current_time + 'transformer_testing')
        '''


class TransformerParams:
    def __init__(self):
        self.learning_rate = 1e-4
        self.batch_size = 32
        self.vocab_size = VOCAB_SIZE
        self.num_heads = 4
        self.max_context = 512
        self.embedding_dim = 512
        self.ffn_expansion = 4
        self.is_training = True
        self.num_blocks = 1
