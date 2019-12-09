from constants import VOCAB_SIZE

BATCH_SIZE = 32
LSTM_HIDDEN_SIZE = 128
delimiter_token = VOCAB_SIZE - 1  # == newline_character; which is out-of-vocabulary. vocab starts at 0, so do not +1
NUM_EPOCHS = 1
TRAINING_STEPS = 10000