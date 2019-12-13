import os
from constants import VOCAB_SIZE
from datetime import datetime as dt
current_time = dt.now().strftime('%Y%m%d_%H_%M-')

BATCH_SIZE = 1
EMBEDDING_SIZE = 512
LSTM_HIDDEN_SIZE = 1024
NUM_EPOCHS = 50
NUM_EXAMPLES = 666666*3
p_test = .2
EXPERIMENT_DIR = os.path.join('experiment_results', current_time + 'arithmetic__add_or_sub_ALL_DIFFICULTY_batch256_lstm512_epochs50')
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
