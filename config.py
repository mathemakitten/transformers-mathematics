import os
from constants import VOCAB_SIZE, QUESTION_MAX_LENGTH, ANSWER_MAX_LENGTH
from datetime import datetime as dt

current_time = dt.now().strftime('%Y%m%d_%H_%M-')

class LSTMParams:
    def __init__(self):
        self.experiment_dir = os.path.join('experiment_results', current_time + 'lstm_run')
        self.tb_logdir = os.path.join(self.experiment_dir, 'tensorboard')

        self.batch_size = 256
        self.num_epochs = 10
        self.num_examples = 666666*3
        self.num_training_batches = int(self.num_examples/self.batch_size*(1-self.p_test))
        self.p_test = 0.2
        self.learning_rate = 1e-4

        self.embedding_size = 512
        self.lstm_hidden_size = 512


class TransformerParams:
    def __init__(self):
        self.experiment_dir = os.path.join('experiment_results', current_time + 'transformer_testing')
        self.checkpoint_dir = os.path.join(self.experiment_dir, 'checkpoints')
        self.tb_logdir = os.path.join(self.experiment_dir, 'tensorboard')

        self.questions_max_length = QUESTION_MAX_LENGTH
        self.answer_max_length = ANSWER_MAX_LENGTH

        self.batch_size = 32
        self.vocab_size = VOCAB_SIZE
        self.num_epochs = 1
        self.num_examples = 10000
        self.p_test = 0.2
        self.num_training_batches = int(self.num_examples/self.batch_size*(1-self.p_test))
        self.learning_rate = 1e-4

        self.max_context = QUESTION_MAX_LENGTH
        self.is_training = True
        self.embedding_dim = 512
        self.num_heads = 4
        self.d_model = 256
        self.dff = 512
        self.ffn_expansion = 4
        self.dropout = 0
        self.attention_dropout = 0
        self.num_layers = 2  # Todo: split this into encoder and decoder layers