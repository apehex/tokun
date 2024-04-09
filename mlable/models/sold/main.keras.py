"""Tensorflow port of the tutorial by Andrej Karpathy: https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/"""

import datetime
import functools
import os

import tensorflow as tf
import tensorflow_datasets as tfds

import mlable.models.tokun.datasets.mlqa as _mlqa
import mlable.models.tokun.datasets.pipeline as _mmad
import mlable.tensorflow.layers as _mtl
import mlable.tensorflow.optimizers as _mto
import mlable.tensorflow.sampling as _sam
import mlable.tensorflow.summary as _sum

# META ########################################################################

N_CONTEXT_DIM = 16 # C
N_TOKEN_DIM = 4 # G
N_ENCODING_DIM = 256 # U
N_EMBEDDING_DIM = N_ENCODING_DIM # E
N_LATENT_DIM = N_EMBEDDING_DIM # L

N_EPOCHS = 2
N_EPOCHS_RAMPUP = 4
N_EPOCHS_SUSTAIN = 0

N_BATCH = 64

N_SAMPLE = 256

R_MIN = 0.0001
R_MAX = 0.001
R_EXP = .8

VERSION = 'tokun-keras-660k'

# DATA ########################################################################

DATA_TRAIN, DATA_TEST = tfds.load('mlqa/en', split=['test', 'validation'], shuffle_files=True, data_dir='~/.cache/tensorflow/')

BATCH_TRAIN = iter(DATA_TRAIN.batch(512))
BATCH_TEST = iter(DATA_TEST.batch(512))

# SPLIT #######################################################################

X_TRAIN = _mmad.tokenize(text=_mlqa.preprocess(next(BATCH_TRAIN)['context']))
X_TEST = _mmad.tokenize(text=_mlqa.preprocess(next(BATCH_TEST)['context']))

Y_TRAIN = tf.one_hot(indices=X_TRAIN, depth=N_ENCODING_DIM, dtype=tf.dtypes.float32) # one-hot encoding of x as y, to compare with the output probabilities
Y_TEST = tf.one_hot(indices=X_TEST, depth=N_ENCODING_DIM, dtype=tf.dtypes.float32) # idem

# MODEL #######################################################################

class Encoder(tf.keras.models.Model):
    def __init__(self, token_dim: int, encoding_dim: int, embedding_dim: int, latent_dim: int, batch_dim: int=None, **kwargs) -> None:
        super(Encoder, self).__init__(**kwargs)
        self._encoder = tf.keras.Sequential([
            # tf.keras.Input(shape=(), batch_size=batch_dim * token_dim, name='input'),
            # _mtl.Merge(input_axis=0, output_axis=1, factor=token_dim, insert=True, name='group'), # (B * G,) => (B, G)
            tf.keras.Input(shape=(token_dim,), batch_size=batch_dim, name='input'),
            tf.keras.layers.Embedding(input_dim=encoding_dim, output_dim=embedding_dim, embeddings_initializer='he_normal', name='embedding'), # (B, G) => (B, G, E)
            _mtl.PositionalEmbedding(input_axis=1, output_axis=-1, name='position'), # (B, G, E) + (1, G, E)
            tf.keras.layers.Flatten(name='flatten'), # (B, G, E) => (B, G * E)
            tf.keras.layers.Dense(units=latent_dim, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', name='head')]) # (B, G * E) => (B, L), here L = E

    def call(self, x: tf.Tensor) -> tf.Tensor:
        return self._encoder(x)

class Decoder(tf.keras.models.Model):
    def __init__(self, token_dim: int, encoding_dim: int, embedding_dim: int, latent_dim: int, batch_dim: int=None, **kwargs) -> None:
        super(Decoder, self).__init__(**kwargs)
        self._decoder = tf.keras.Sequential([
            tf.keras.Input(shape=(latent_dim,), batch_size=batch_dim, name='input'),
            tf.keras.layers.Dense(units=token_dim * embedding_dim, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', name='head'), # (B, L) => (B, G * E), here L = E
            tf.keras.layers.Reshape(target_shape=(token_dim, embedding_dim), name='reshape'), # (B, G * E) => (B, G, E)
            tf.keras.layers.Dense(units=encoding_dim, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', name='feet'), # (B, G, E) => (B, G, U), here U = E
            # _mtl.Merge(input_axis=1, output_axis=0, factor=token_dim, insert=False, name='split'), # (B, G, E) => (B * G, E)
            tf.keras.layers.Softmax(axis=-1, name='softmax')])

    def call(self, x: tf.Tensor) -> tf.Tensor:
        return self._decoder(x)

class AutoEncoder(tf.keras.models.Model):
    def __init__(self, token_dim: int, encoding_dim: int, embedding_dim: int, latent_dim: int, batch_dim: int=None, **kwargs) -> None:
        super(AutoEncoder, self).__init__(**kwargs)
        self._encoder = Encoder(token_dim=token_dim, encoding_dim=encoding_dim, embedding_dim=embedding_dim, latent_dim=latent_dim, batch_dim=None)
        self._decoder = Decoder(token_dim=token_dim, encoding_dim=encoding_dim, embedding_dim=embedding_dim, latent_dim=latent_dim, batch_dim=None)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        return self._decoder(self._encoder(x))

# INIT ########################################################################

# (B, 4) => (B, 4, 256) => (B, 1024) => (B, 256)
# (B, 256) => (B, 1024) => (B, 4, 256) => (B, 4, 256) => (B, 4, 256)
MODEL = AutoEncoder(token_dim=N_TOKEN_DIM, encoding_dim=N_ENCODING_DIM, embedding_dim=N_EMBEDDING_DIM, latent_dim=N_LATENT_DIM, batch_dim=N_BATCH)

# compile
MODEL.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=R_MAX),
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False, label_smoothing=0., axis=-1, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE, name='loss'),
    metrics=['accuracy'])

# SAVE ########################################################################

# log path
LOGPATH = os.path.join('.logs/', VERSION, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
SUMMARY = tf.summary.create_file_writer(LOGPATH)

# called during training
tb_callback = tf.keras.callbacks.TensorBoard(log_dir=LOGPATH)

# LEARNING RATE ###############################################################

lr_callback = tf.keras.callbacks.LearningRateScheduler(functools.partial(_mto.learning_rate_hokusai, lr_min=R_MIN, lr_max=R_MAX, lr_exp=R_EXP, rampup=N_EPOCHS_RAMPUP, sustain=N_EPOCHS_SUSTAIN), verbose=True)

# TRAIN #######################################################################

TRAINING_HISTORY = MODEL.fit(
    x=X_TRAIN,
    y=Y_TRAIN,
    batch_size=N_BATCH * N_TOKEN_DIM,
    epochs=N_EPOCHS,
    validation_split=None,
    validation_data=(X_TEST, Y_TEST),
    validation_freq=[1, N_EPOCHS],
    verbose=2,
    callbacks=[lr_callback, tb_callback])

# SAMPLE ######################################################################

# sample = functools.partial(_sam.sample, model=MODEL, context=N_CONTEXT_DIM, depth=N_ENCODING_DIM, length=N_SAMPLE)

# VIZ #########################################################################

# _sum.save_model_histograms(model=MODEL, epoch=N_EPOCHS, summary=SUMMARY)
