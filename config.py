from constants import VOCAB_SIZE

BATCH_SIZE = 32
EMBEDDING_SIZE = 32
LSTM_HIDDEN_SIZE = 128
delimiter_token = VOCAB_SIZE - 1  # == newline_character; which is out-of-vocabulary. vocab starts at 0, so do not +1
NUM_EPOCHS = 1000
NUM_EXAMPLES = 10
p_test = 0.2