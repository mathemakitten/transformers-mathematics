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


class TransformerParams:
    def __init__(self):
        self.TM_learning_rate = 1e-4
        self.batch_size = 32
        self.vocab_size = 8185
        self.num_heads = 4
        self.max_context = 1024
        self.embedding_dim = 512
        self.ffn_expansion = 4
        self.is_training = True
        self.num_blocks = 1
