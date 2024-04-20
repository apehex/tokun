"""Tensorflow port of the tutorial by Andrej Karpathy: https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/"""

import datetime
import functools
import os

import tensorflow as tf
import tensorflow_datasets as tfds

import mlable.models.tokun.layers as _mmtl
import mlable.models.tokun.pipeline as _mmtp
import mlable.tensorflow.io as _mti
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

N_EPOCHS = 24
N_EPOCHS_RAMPUP = 12
N_EPOCHS_SUSTAIN = 0

N_BATCH = 128 # number of samples per batch
N_SAMPLE = 128 # number of characters per sample (=> N_TOKEN_DIM * N_SAMPLE integers per sample)

R_MIN = 0.0001
R_MAX = 0.001
R_EXP = .8

VERSION = 'tokun-1-keras-660k'

# DATA ########################################################################

# DATA_TRAIN, DATA_TEST = tfds.load('mlqadd', split=['train', 'test'], as_supervised=True, shuffle_files=True, data_dir='~/.cache/tensorflow/', builder_kwargs={'train_lang': ['en'], 'test_lang': ['es']})
LANG = ['ar', 'de', 'en', 'es', 'hi', 'vi', 'zh']
DATA = {__l: tfds.load('mlqa/' + __l, split='test', as_supervised=False, shuffle_files=True, data_dir='~/.cache/tensorflow/', batch_size=N_BATCH) for __l in LANG}

# MODEL #######################################################################

class Encoder(tf.keras.models.Model):
    def __init__(self, token_dim: int, encoding_dim: int, embedding_dim: int, latent_dim: int, batch_dim: int=None, **kwargs) -> None:
        super(Encoder, self).__init__(**kwargs)
        self._encoder = tf.keras.Sequential([
            tf.keras.Input(shape=(encoding_dim,), batch_size=batch_dim, name='input'), # (B * G * G, U)
            tf.keras.layers.Dense(units=embedding_dim, activation=None, use_bias=False, kernel_initializer='glorot_uniform', bias_initializer=None, name='embed-1'), # (B * G * G, U) => (B * G * G, E)
            _mmtl.TokenizeBlock(left_axis=-2, right_axis=-1, token_dim=token_dim, latent_dim=latent_dim, name='tokenize-4'), # (B * G * G, E) => (B * G, E)
            _mmtl.TokenizeBlock(left_axis=-2, right_axis=-1, token_dim=token_dim, latent_dim=latent_dim, name='tokenize-4-4'),]) # (B * G, E) => (B, E)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        return self._encoder(x)

class Decoder(tf.keras.models.Model):
    def __init__(self, token_dim: int, encoding_dim: int, embedding_dim: int, latent_dim: int, batch_dim: int=None, **kwargs) -> None:
        super(Decoder, self).__init__(**kwargs)
        self._decoder = tf.keras.Sequential([
            tf.keras.Input(shape=(latent_dim,), batch_size=batch_dim, name='input'), # (B, E)
            _mmtl.DetokenizeBlock(token_dim=token_dim, embedding_dim=embedding_dim, name='detokenize-4-4'), # (B, E) => (B * G, E)
            _mmtl.DetokenizeBlock(token_dim=token_dim, embedding_dim=embedding_dim, name='detokenize-4'), # (B * G, E) => (B * G * G, E)
            _mmtl.HeadBlock(encoding_dim=encoding_dim, name='project-head')]) # (B * G * G, E) => (B * G * G, U)

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

# PREPROCESS ##################################################################

DATA = {__l: _mmtp.preprocess(dataset=__d, key='context', layer_count=2, group_size=N_TOKEN_DIM, sample_size=N_SAMPLE, flatten=True) for __l, __d in DATA.items()}

# TRAIN #######################################################################

TRAINING_HISTORY = MODEL.fit(
    x=DATA['en'],
    batch_size=N_BATCH,
    epochs=N_EPOCHS,
    validation_split=None,
    validation_data=DATA['zh'], # full of glyphs
    validation_freq=[1, N_EPOCHS],
    verbose=2,
    callbacks=[lr_callback, tb_callback])

# SAMPLES #####################################################################

__i = iter(DATA['de']) # iterate over batches of samples
__x = next(__i)[0][:2 * N_TOKEN_DIM * N_SAMPLE] # take two samples from the batch
__o = MODEL.predict(__x)

print(_mmtp.postprocess(__x))
print(_mmtp.postprocess(__o))

# DATAVIZ #####################################################################

__characters = _mmtp.chunk(seq=_mmtp.postprocess(__x), size=1)
__tokens = _mmtp.chunk(seq=_mmtp.postprocess(__x), size=4)

__character_embeddings = MODEL._encoder._encoder.layers[0](MODEL._encoder._encoder.layers[1](__x))
__token_embeddings = MODEL._encoder.predict(__x)

_mti.write(data=__characters, path='./metadata.characters.tsv', tsv=False)
_mti.write(data=__character_embeddings, path='./embeddings.characters.tsv', tsv=True)

_mti.write(data=__tokens, path='./metadata.tokens.tsv', tsv=False)
_mti.write(data=__token_embeddings, path='./embeddings.tokens.tsv', tsv=True)
