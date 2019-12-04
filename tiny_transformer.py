""" Tiny transformer implementation in TF 2.0 """

import os
import numpy as np
import argparse
import time
from utils import get_logger

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# Argparse setup
parser = argparse.ArgumentParser(description='tiny transformer with linear attention',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--run_name', metavar='PATH', type=str, default='run', help='Directory name for run.')
parser.add_argument('--eager', metavar='eager_mode', type=bool, default=False, help='Eager mode on, else Autograph')
parser.add_argument('--gpu', metavar='which_gpu', type=int, default=1, help='Which GPU to run on')

args = parser.parse_args()

# Tensorflow env setup
import tensorflow as tf
tf.config.experimental_run_functions_eagerly(args.eager)

# Logger setup
logger = get_logger('TRAIN', args.run_name)

# Constants + params
tf.random.set_seed(1)


class Params:  # TODO move this to not be... here
    def __init__(self):
        self.learning_rate = 1e-4
        self.batch_size = 32
        self.vocab_size = 8185
        self.num_heads = 4
        self.max_context = 1024
        self.embedding_dim = 512
        self.ffn_expansion = 4
        self.is_training = True
        self.num_blocks = 1
        self.logdir = 'tensorboard/' + args.run_name


params = Params()

def dict_to_tuple(example):
    return (tf.cast(example['text'][0:params.max_context], tf.int32), tf.cast(example['label'], tf.float32))


# Create tf.Data object for feeding input data
mode = 'train'
dataset, info = tfds.load(name='imdb_reviews/subwords8k', with_info=True, split='train')

input_data = dataset.take(50000).shuffle(1000).map(dict_to_tuple)\
    .padded_batch(batch_size=params.batch_size, padded_shapes=([None], []), padding_values=(0, 0.0))\
    .repeat(10).prefetch(buffer_size=1)


class PositionalEncodingLayer(tf.keras.layers.Layer):
    def __init__(self, max_position, d_model, name='PositionalEncodingLayer'):
        super(PositionalEncodingLayer, self).__init__()
        # Create a table of all possible positional encodings the model will see
        self.pos_encoding = self.compute_positional_encoding(max_position, d_model, name=name)

    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
        return pos * angle_rates

    def compute_positional_encoding(self, position, d_model, name):
        angle_rads = self.get_angles(np.arange(position)[:, np.newaxis],
                                     np.arange(d_model)[np.newaxis, :],
                                     d_model)
        # apply sin to even indices in the array; 2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        # apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32, name=name)

    def call(self, seq_len):
        # Select relevant positions up to the sequence length
        return self.pos_encoding[:, :seq_len, :]



class mlp(tf.keras.layers.Layer):

    def __init__(self, neurons):
        super(mlp, self).__init__()
        self.neurons = int(neurons)
        self.ffn_dim = params.embedding_dim  # still all the same dims...
        self.hidden1 = tf.keras.layers.Dense(units=self.neurons, activation='relu', name='mlp_h1')
        self.hidden2 = tf.keras.layers.Dense(units=self.ffn_dim, name='mlp_h2')

    def call(self, x):
        h = self.hidden1(x)
        h2 = self.hidden2(h)
        return h2


class decoder_block(tf.keras.layers.Layer):
    # TODO: attention dropout

    def __init__(self, num_heads, scope='decoder'):
        super(decoder_block, self).__init__()

        self.projections = [tf.keras.layers.Dense(params.embedding_dim * 3 // params.num_heads) for _ in range(num_heads)]
        self.multihead_attn = tf.keras.layers.Dense(params.embedding_dim, name='multihead_attn')
        self.layernorm1 = tf.keras.layers.LayerNormalization(name='layernorm1')
        self.layernorm2 = tf.keras.layers.LayerNormalization(name='layernorm2')
        self.scope = scope
        self.mlp = mlp(neurons=params.embedding_dim * params.ffn_expansion)

    def call(self, x):

        residual = x
        num_heads = params.num_heads

        all_heads = []
        for i in range(num_heads):
            d = self.projections[i](x)
            q, k, v = tf.split(d, num_or_size_splits=3, axis=-1)
            dim_k = tf.shape(k)[1]
            qk_t = tf.matmul(q, k, transpose_b=True) * tf.math.rsqrt(tf.cast(dim_k, dtype=tf.float32))

            if args.linear_attention:
                head = tf.matmul(qk_t, v)
            else:
                head = tf.matmul(tf.nn.softmax(qk_t), v)  # regular softmax scaled dot product attention

            all_heads.append(head)

        all_heads_tensor = tf.concat(all_heads, axis=-1)
        multihead_attn = self.multihead_attn(all_heads_tensor)

        x = multihead_attn
        x = x + residual
        x = self.layernorm1(x)

        residual2 = x
        x = self.mlp(x)
        x = x + residual2
        x = self.layernorm2(x)

        #print("DIM FFN is the same as output of this layer which is {}".format(x.shape))

        return x


class TinyTransformer(tf.keras.Model):
    def __init__(self, params):
        super(TinyTransformer, self).__init__(self)
        self.hparams = params
        self.embedding_layer = tf.keras.layers.Embedding(params.vocab_size, params.embedding_dim, name='embedding_layer')
        self.positional_embeddings = PositionalEncodingLayer(max_position=params.max_context, d_model=params.embedding_dim)
        self.optimizer = tf.keras.optimizers.Adam(lr=params.learning_rate)

        self.blocks = []
        for i in range(params.num_blocks):  # 4x more decoder blocks
            self.blocks.append(decoder_block(num_heads=params.num_heads, scope=str(i)))

        self.hidden = tf.keras.layers.Dense(units=params.embedding_dim, activation='relu', name='final_hidden')
        self.final_dense = tf.keras.layers.Dense(units=1, name='final_dense')


    def call(self, inputs):

        # Embed inputs
        embedded_inputs = self.embedding_layer(inputs)
        embedded_inputs_with_positions = embedded_inputs + self.positional_embeddings(tf.shape(inputs)[1])

        # From embeddings
        x = embedded_inputs_with_positions

        for block in self.blocks:
            x = block(x)

        x = x[:, 0]  # this is better than the reduce_mean according to our friend Bert
        #x = tf.reduce_mean(x, axis=1)  # TODO reduce better

        x = self.hidden(x)

        logits = self.final_dense(x)
        outputs = tf.nn.sigmoid(logits)

        return logits, outputs

    def compute_loss(self, logits, labels):
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits[:, 0]))
        return loss

    def tensorboard_profile(self, writer, logdir):
        with writer.as_default():
            tf.summary.trace_export(
                name="Trace_loss",
                step=0,
                profiler_outdir=logdir)


    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None], dtype=tf.int32), tf.TensorSpec(shape=[None, ], dtype=tf.float32)])
    def train_step(self, inputs, targets):

        with tf.GradientTape() as tape:
            logits, outputs = self.call(inputs=inputs)
            loss = self.compute_loss(logits, targets)

        grads = tape.gradient(loss, self.trainable_variables)

        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        updates = [tf.reduce_mean(tf.abs(g)) / tf.reduce_mean(tf.abs(v)) if g is not None else None for g, v in
                   zip(grads, self.trainable_variables)]
        updates = {v.name: u for v, u in zip(self.trainable_variables, updates) if u is not None}

        return loss, updates

    def train(self, input_data):

        writer = tf.summary.create_file_writer(params.logdir)

        for i, data in enumerate(input_data):

            if i == 0:
                tf.summary.trace_on(graph=True, profiler=True)

            # Run a single training step
            start_time = time.time()
            loss, updates = self.train_step(data[0], data[1])
            time_per_step = time.time() - start_time

            if i == 0:
                self.tensorboard_profile(writer, params.logdir)
                tf.summary.trace_off()

            if i % 100 == 0:
                logger.info("Step: {} - Loss: {} - Time per step: {}".format(i, loss, time_per_step))
                with writer.as_default():
                    tf.summary.scalar(f"Losses/total_loss", loss, step=i)

                    for variable in self.trainable_variables:
                        tf.summary.histogram("Weights/{}".format(variable.name), variable, step=i)

                    for layer, update in updates.items():
                        tf.summary.scalar("Updates/{}".format(layer), update, step=i)

                    mean_updates = tf.reduce_mean(list(updates.values()))
                    max_updates = tf.reduce_max(list(updates.values()))
                    tf.summary.scalar("Mean_Max_Updates/Mean_updates", mean_updates, step=i)
                    tf.summary.scalar("Mean_Max_Updates/Max_updates", max_updates, step=i)

                    writer.flush()


model = TinyTransformer(params)
logger.info("Model training")
model.train(input_data)
logger.info("Model train completed!")

