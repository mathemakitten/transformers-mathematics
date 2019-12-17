import tensorflow as tf
import numpy as np
import os
import argparse
from constants import ANSWER_MAX_LENGTH
from preprocessing import idx2char  # TODO cache questions/answers_encoded as .npy files
from config import *
from utils import get_logger
import time
from constants import QUESTION_MAX_LENGTH
# from lstm import inference_step, inference, get_validation_metrics, get_accuracy

# logger = get_logger('validation_log')
# parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# parser.add_argument('--eager', metavar='eager_mode', type=bool, default=True, help='Eager mode on, else Autograph')
# parser.add_argument('--gpu_id', metavar='gpu_id', type=str, default="1", help='The selected GPU to use, default 1')
# args = parser.parse_args()
#
# tf.config.experimental_run_functions_eagerly(args.eager)
# os.environ['CUDA_VISIBLE_DEVICES'] = "3"
#
# # load pre-padded data
# dataset_id = '_all_data_ever'
# questions_encoded = np.array(np.load('cache/questions_encoded_padded_{}.npy'.format(dataset_id)))
# answers_encoded = np.array(np.load('cache/answers_encoded_padded_{}.npy'.format(dataset_id)))
#
# params = TransformerParams()
#
# dataset = tf.data.Dataset.from_tensor_slices((questions_encoded, answers_encoded))
# input_data = dataset.take(params.num_examples).shuffle(questions_encoded.shape[0]).batch(params.batch_size) \
#             .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
# train_data = input_data.take(params.num_training_batches).repeat(params.num_epochs)
# valid_data = input_data.skip(params.num_training_batches)


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
      q: query shape == (..., seq_len_q, depth)
      k: key shape == (..., seq_len_k, depth)
      v: value shape == (..., seq_len_v, depth_v)
      mask: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
      output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

        # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights



def point_wise_feed_forward_network(d_model, dff):
  return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
  ])


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training,
             look_ahead_mask, padding_mask):
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

        return out3, attn_weights_block1, attn_weights_block2


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, maximum_position_encoding, rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, self.d_model)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]

        # adding embedding and position encoding.
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)


class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
                 maximum_position_encoding, rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training,
             look_ahead_mask, padding_mask):
        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                                   look_ahead_mask, padding_mask)

            attention_weights['decoder_layer{}_block1'.format(i + 1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i + 1)] = block2

        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights


class Transformer(tf.keras.Model):
    def __init__(self, params):
        super(Transformer, self).__init__()

        self.learning_rate = CustomSchedule(params.d_model)
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        num_layers = params.num_layers
        d_model = params.d_model
        num_heads = params.num_heads
        dff = params.dff
        input_vocab_size = params.vocab_size
        target_vocab_size = params.vocab_size
        pe_input = params.questions_max_length
        pe_target = params.answer_max_length
        attention_dropout = params.attention_dropout

        self.encoder = Encoder(num_layers, d_model, num_heads, dff, input_vocab_size, pe_input, attention_dropout)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, target_vocab_size, pe_target, attention_dropout)
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inp, tar, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):

        enc_output = self.encoder(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)
        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(tar, enc_output, training, look_ahead_mask, dec_padding_mask)
        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

        return final_output, attention_weights

    def loss_function(self, real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = self.loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_mean(loss_)

    @tf.function(input_signature=[tf.TensorSpec(shape=(None, None), dtype=tf.int64),
                                  tf.TensorSpec(shape=(None, None), dtype=tf.int64), ])
    def train_step(self, inputs, targets):
        targets_with_start_token = targets[:, :-1]
        targets_no_start_token = targets[:, 1:]

        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inputs, targets_with_start_token)

        with tf.GradientTape() as tape:
            predictions, _ = self.call(inputs, targets_with_start_token,
                                   True,
                                   enc_padding_mask,
                                   combined_mask,
                                   dec_padding_mask)
            loss = self.loss_function(targets_no_start_token, predictions)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.train_loss(loss)
        # train_accuracy(tar_real, predictions)
        return predictions

    def train(self, params, train_data, valid_data, logger):
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

        checkpoint_path = params.checkpoint_dir
        ckpt = tf.train.Checkpoint(transformer=self, optimizer=self.optimizer)
        ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
        # if a checkpoint exists, restore the latest checkpoint.
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print('Latest checkpoint restored!!')

        valid_loss_list = []
        best_loss = 0

        for epoch in range(params.num_epochs):
            start = time.time()
            self.train_loss.reset_states()

            accuracy_list = []
            for batch, data in enumerate(train_data):
                inp = data[0]
                tar = data[1]
                predictions = self.train_step(inp, tar)

                first_padding_positions = tf.argmax(
                    tf.cast(tf.equal(tf.cast(tf.zeros(tar.shape), dtype=tf.float32), tf.cast(tar, dtype=tf.float32)),
                            tf.float32), axis=1)
                preds = tf.argmax(predictions, axis=-1)

                padding_mask = tf.sequence_mask(lengths=first_padding_positions, maxlen=ANSWER_MAX_LENGTH,
                                                dtype=tf.int64)
                preds_to_compare = preds * padding_mask
                targets_to_compare = tar[:, 1:] * padding_mask

                # Compare row-by-row for exact match between preds / true target sequences
                correct_pred_mask = tf.reduce_all(tf.equal(preds_to_compare, targets_to_compare), axis=1)
                accuracy = tf.reduce_sum(tf.cast(correct_pred_mask, dtype=tf.int32)) / tf.shape(correct_pred_mask)[0]
                accuracy_list.append(accuracy)

                if batch % 50 == 0:
                    print('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                        epoch + 1, batch, self.train_loss.result(), np.mean(accuracy_list[-50:])))

            # todo
            # if (epoch + 1) % 5 == 0:
            #     valid_loss, valid_acc = get_validation_metrics(valid_data, self)
            #     valid_loss_list.append(valid_loss)
            #     if valid_loss < best_loss:
            #         best_loss = valid_loss
            #         logger.info(f'Saving on batch {batch}')
            #         logger.info(f'New best validation loss: {best_loss}')
            #         ckpt_save_path = ckpt_manager.save()
            #         print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
            #                                                             ckpt_save_path))
            #
            # print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1,
            #                                                     self.train_loss.result(),
            #                                                     np.mean(accuracy_list[-50:])))
            #
            # print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))
            #
            #
            # if all([valid_loss < best_loss for valid_loss in valid_loss_list[-5:]]):
            #     break


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

def create_masks(inp, tar):
    # Encoder padding mask
    enc_padding_mask = create_padding_mask(inp)

    # Used in the 2nd attention block in the decoder.
    # This padding mask is used to mask the encoder outputs.
    dec_padding_mask = create_padding_mask(inp)

    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by
    # the decoder.
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask

def evaluate(inp_sentence):
    start_token = [tokenizer_pt.vocab_size]
    end_token = [tokenizer_pt.vocab_size + 1]

    # inp sentence is portuguese, hence adding the start and end token
    inp_sentence = start_token + tokenizer_pt.encode(inp_sentence) + end_token
    encoder_input = tf.expand_dims(inp_sentence, 0)

    # as the target is english, the first word to the transformer should be the
    # english start token.
    decoder_input = [tokenizer_en.vocab_size]
    output = tf.expand_dims(decoder_input, 0)

    for i in range(MAX_LENGTH):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
            encoder_input, output)
        # predictions.shape == (batch_size, seq_len, vocab_size)
        predictions, attention_weights = transformer(encoder_input,
                                                     output,
                                                     False,
                                                     enc_padding_mask,
                                                     combined_mask,
                                                     dec_padding_mask)

        # select the last word from the seq_len dimension
        predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        # return the result if the predicted_id is equal to the end token
        if predicted_id == tokenizer_en.vocab_size + 1:
            return tf.squeeze(output, axis=0), attention_weights

        # concatentate the predicted_id to the output which is given to the decoder
        # as its input.
        output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0), attention_weights

'''
def translate(sentence, plot=''):
    result, attention_weights = evaluate(sentence)

    predicted_sentence = tokenizer_en.decode([i for i in result
                                              if i < tokenizer_en.vocab_size])

    print('Input: {}'.format(sentence))
    print('Predicted translation: {}'.format(predicted_sentence))

    if plot:
        plot_attention_weights(attention_weights, sentence, result, plot)
'''


if __name__ == '__main__':
    # learning_rate = CustomSchedule(d_model)
    # optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
    #                                      epsilon=1e-9)
    #
    # loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    #
    # train_loss = tf.keras.metrics.Mean(name='train_loss')
    #
    # model = Transformer(num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size,
    #                           pe_input=QUESTION_MAX_LENGTH,#input_vocab_size,
    #                           pe_target=ANSWER_MAX_LENGTH,#target_vocab_size,
    #                           rate=dropout_rate)
    #
    # checkpoint_path = "./checkpoints/train"
    #
    # ckpt = tf.train.Checkpoint(transformer=model,
    #                            optimizer=optimizer)
    #
    # ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
    #
    # # if a checkpoint exists, restore the latest checkpoint.
    # if ckpt_manager.latest_checkpoint:
    #     ckpt.restore(ckpt_manager.latest_checkpoint)
    #     print('Latest checkpoint restored!!')
    #
    # valid_loss_list = []
    # best_loss = 0
    #
    # for epoch in range(EPOCHS):
    #     start = time.time()
    #     train_loss.reset_states()
    #
    #     accuracy_list = []
    #     for batch, data in enumerate(train_data):
    #         inp = data[0]
    #         tar = data[1]
    #         predictions = train_step(inp, tar)
    #
    #         first_padding_positions = tf.argmax(tf.cast(tf.equal(tf.cast(tf.zeros(tar.shape), dtype=tf.float32), tf.cast(tar, dtype=tf.float32)), tf.float32), axis=1)
    #         preds = tf.argmax(predictions, axis=-1)
    #
    #         padding_mask = tf.sequence_mask(lengths=first_padding_positions, maxlen=ANSWER_MAX_LENGTH, dtype=tf.int64)
    #         preds_to_compare = preds * padding_mask
    #         targets_to_compare = tar[:, 1:] * padding_mask
    #
    #         # Compare row-by-row for exact match between preds / true target sequences
    #         correct_pred_mask = tf.reduce_all(tf.equal(preds_to_compare, targets_to_compare), axis=1)
    #         accuracy = tf.reduce_sum(tf.cast(correct_pred_mask, dtype=tf.int32))/tf.shape(correct_pred_mask)[0]
    #         accuracy_list.append(accuracy)
    #
    #         if batch % 50 == 0:
    #             print('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
    #                 epoch + 1, batch, train_loss.result(), np.mean(accuracy_list[-50:])))
    #
    #     if (epoch + 1) % 5 == 0:
    #         valid_loss, valid_acc = get_validation_metrics(valid_data, model)
    #         valid_loss_list.append(valid_loss)
    #         if valid_loss < best_loss:
    #             best_loss = valid_loss
    #             logger.info(f'Saving on batch {batch}')
    #             logger.info(f'New best validation loss: {best_loss}')
    #             ckpt_save_path = ckpt_manager.save()
    #             print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
    #                                                             ckpt_save_path))
    #
    #     print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1,
    #                                                         train_loss.result(),
    #                                                         np.mean(accuracy_list[-50:])))
    #
    #     print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))
    #
    #     if all([valid_loss < best_loss for valid_loss in valid_loss_list[-5:]]):
    #         break
    #

    print("empty main!!")


'''
def plot_attention_weights(attention, sentence, result, layer):
    fig = plt.figure(figsize=(16, 8))

    sentence = tokenizer_pt.encode(sentence)

    attention = tf.squeeze(attention[layer], axis=0)

    for head in range(attention.shape[0]):
        ax = fig.add_subplot(2, 4, head + 1)

        # plot the attention weights
        ax.matshow(attention[head][:-1, :], cmap='viridis')

        fontdict = {'fontsize': 10}

        ax.set_xticks(range(len(sentence) + 2))
        ax.set_yticks(range(len(result)))

        ax.set_ylim(len(result) - 1.5, -0.5)

        ax.set_xticklabels(
            ['<start>'] + [tokenizer_pt.decode([i]) for i in sentence] + ['<end>'],
            fontdict=fontdict, rotation=90)

        ax.set_yticklabels([tokenizer_en.decode([i]) for i in result
                            if i < tokenizer_en.vocab_size],
                           fontdict=fontdict)

        ax.set_xlabel('Head {}'.format(head + 1))

    plt.tight_layout()
    plt.show()
'''

