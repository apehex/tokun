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

N_DEPTH = 3 # D
N_TOKEN_DIM = 4 # G
N_ENCODING_DIM = 256 # U
N_EMBEDDING_DIM = N_ENCODING_DIM # E
N_LATENT_DIM = N_EMBEDDING_DIM # L

N_EPOCHS = 8
N_EPOCHS_RAMPUP = 0
N_EPOCHS_SUSTAIN = 0

N_BATCH = 128 # number of samples per batch
N_SAMPLE = 128 # number of characters per sample (=> N_TOKEN_DIM * N_SAMPLE integers per sample)

R_MIN = 0.00001
R_MAX = 0.0001
R_EXP = .8

VERSION = 'tokun-4x4-keras-1M700K'

# LOG #########################################################################

LOGPATH = os.path.join('.logs/', VERSION, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
SUMMARY = tf.summary.create_file_writer(LOGPATH)

# DATA ########################################################################

LANG = ['ar', 'de', 'en', 'es', 'hi', 'vi', 'zh']
TRAIN = {__l: tfds.load('mlqa/' + __l, split='test', as_supervised=False, shuffle_files=True, data_dir='~/.cache/tensorflow/', batch_size=N_BATCH) for __l in LANG}
TEST = {__l: tfds.load('mlqa/' + __l, split='validation', as_supervised=False, shuffle_files=True, data_dir='~/.cache/tensorflow/', batch_size=N_BATCH) for __l in LANG}

# PREPROCESS ##################################################################

# B = 128, T = 4, S = 128, E = 256
PIPELINE = [
    # offset by 1 to 15 character => (B, 1) bytes
    (functools.partial(_mmtp.offset, ticks=1, layer=1, unit=N_TOKEN_DIM), False), # double the dataset volume: (all samples with offset 0) + (offset 1)
    (functools.partial(_mmtp.offset, ticks=2, layer=1, unit=N_TOKEN_DIM), False), # (offsets 0 and 1) + (offsets 2 and 3)
    (functools.partial(_mmtp.offset, ticks=4, layer=1, unit=N_TOKEN_DIM), False), # (offsets 0, 1, 2, 3) + (offsets 4, 5, 6, 7)
    (functools.partial(_mmtp.offset, ticks=8, layer=1, unit=N_TOKEN_DIM), False), # (offsets 0, 1, 2, 3, 4, 5, 6, 7) + (offsets 8, 9, 10, 11, 12, 13, 14, 15)
    # tokenize => (B * T * S,) int
    (functools.partial(_mmtp.tokenize, layer_count=N_DEPTH, group_size=N_TOKEN_DIM, sample_size=N_SAMPLE, flatten=True), True),
    # one-hot encoding => (B * T * S, E) int (bool)
    (functools.partial(tf.one_hot, depth=N_ENCODING_DIM, axis=-1), True),
    # replace sample inputs with (inputs, target) for supervised learning
    ((lambda x: (x, x)), True)]

OPERATIONS, REPLACE = zip(*PIPELINE)

TRAIN = {__l: _mmtp.process(dataset=__d, feature='context', pipeline=OPERATIONS, replace=REPLACE) for __l, __d in TRAIN.items()}
TEST = {__l: _mmtp.process(dataset=__d, feature='context', pipeline=OPERATIONS, replace=REPLACE) for __l, __d in TEST.items()}

# MODEL #######################################################################

class Encoder(tf.keras.models.Model):
    def __init__(self, depth: int, token_dim: int, encoding_dim: int, embedding_dim: int, latent_dim: int, batch_dim: int=None, attention: bool=False, normalization: bool=False, **kwargs) -> None:
        super(Encoder, self).__init__(**kwargs)
        self._encoder = tf.keras.Sequential([
            tf.keras.Input(shape=(encoding_dim,), batch_size=batch_dim, name='input'), # (B * G ^ D, U)
            tf.keras.layers.Dense(units=embedding_dim, activation=None, use_bias=False, kernel_initializer='glorot_uniform', bias_initializer=None, name='embed-1'),] # (B * G ^ D, U) => (B * G ^ D, E)
            + [_mmtl.TokenizeBlock(left_axis=-2, right_axis=-1, token_dim=token_dim, latent_dim=latent_dim, attention=attention, normalization=normalization, name='tokenize' + (__i + 1) * '-4') for __i in range(depth)]) # (B * G ^ i, E) => (B * G ^ (i-1), E)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        return self._encoder(x)

class Decoder(tf.keras.models.Model):
    def __init__(self, depth: int, token_dim: int, encoding_dim: int, embedding_dim: int, latent_dim: int, batch_dim: int=None, attention: bool=False, normalization: bool=False, **kwargs) -> None:
        super(Decoder, self).__init__(**kwargs)
        self._decoder = tf.keras.Sequential(
            [tf.keras.Input(shape=(latent_dim,), batch_size=batch_dim, name='input')] # (B, E)
            + [_mmtl.DetokenizeBlock(token_dim=token_dim, embedding_dim=embedding_dim, attention=attention, normalization=normalization, name='detokenize' + (depth - __i) * '-4') for __i in range(depth)] # (B * G ^ i, E) => (B * G ^ (i+1), E)
            + [_mmtl.HeadBlock(encoding_dim=encoding_dim, name='project-head')]) # (B * G ^ D, E) => (B * G ^ D, U)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        return self._decoder(x)

class AutoEncoder(tf.keras.models.Model):
    def __init__(self, depth: int, token_dim: int, encoding_dim: int, embedding_dim: int, latent_dim: int, batch_dim: int=None, attention: bool=False, normalization: bool=False, **kwargs) -> None:
        super(AutoEncoder, self).__init__(**kwargs)
        self._encoder = Encoder(depth=depth, token_dim=token_dim, encoding_dim=encoding_dim, embedding_dim=embedding_dim, latent_dim=latent_dim, batch_dim=batch_dim, attention=attention, normalization=normalization)
        self._decoder = Decoder(depth=depth, token_dim=token_dim, encoding_dim=encoding_dim, embedding_dim=embedding_dim, latent_dim=latent_dim, batch_dim=batch_dim, attention=attention, normalization=normalization)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        return self._decoder(self._encoder(x))

# INIT ########################################################################

MODEL = AutoEncoder(depth=N_DEPTH, token_dim=N_TOKEN_DIM, encoding_dim=N_ENCODING_DIM, embedding_dim=N_EMBEDDING_DIM, latent_dim=N_LATENT_DIM, batch_dim=None, attention=True, normalization=True)

# compile
MODEL.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=R_MAX),
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False, label_smoothing=0., axis=-1, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE, name='loss'),
    metrics=['accuracy'])

# TRAIN #######################################################################

tb_callback = tf.keras.callbacks.TensorBoard(log_dir=LOGPATH)
lr_callback = tf.keras.callbacks.LearningRateScheduler(functools.partial(_mto.learning_rate_hokusai, lr_min=R_MIN, lr_max=R_MAX, lr_exp=R_EXP, rampup=N_EPOCHS_RAMPUP, sustain=N_EPOCHS_SUSTAIN), verbose=True)

# TRAINING_HISTORY = MODEL.fit(
#     x=TRAIN['ar'].concatenate(TRAIN['en']).concatenate(TRAIN['es']).concatenate(TRAIN['de']).concatenate(TRAIN['hi']).concatenate(TRAIN['vi']).concatenate(TRAIN['zh']),
#     batch_size=N_BATCH,
#     epochs=N_EPOCHS,
#     validation_split=None,
#     validation_data=TEST['zh'], # full of glyphs
#     validation_freq=list(range(1, N_EPOCHS + 1, N_EPOCHS // 8)),
#     verbose=2,
#     callbacks=[lr_callback, tb_callback])

# SAMPLES #####################################################################

SAMPLES = {}
TOKENS = {1: {}, 4: {}, 16: {}}
EMBEDDINGS = {1: {}, 4: {}, 16: {}}

for __l in TEST:
    # compute predictions
    __i = iter(TEST[__l]) # iterate over batches of samples
    __x = next(__i)[0] # take input only
    __o = MODEL(__x)
    # sample predictions (inputs, outputs)
    SAMPLES[__l] = (__x, __o)
    # unique 1-tokens (characters)
    TOKENS[1][__l] = _mmtp.chunk(seq=_mmtp.postprocess(__x), size=1, repeats=False)
    # unique 4-tokens
    TOKENS[4][__l] = _mmtp.chunk(seq=_mmtp.postprocess(__x), size=4, repeats=False)
    # unique 4x4-tokens
    TOKENS[16][__l] = _mmtp.chunk(seq=_mmtp.postprocess(__x), size=16, repeats=False)

TOKENS[1]['all'] = list(set(__t for _, __s in TOKENS[1].items() for __t in __s))
TOKENS[4]['all'] = list(set(__t for _, __s in TOKENS[4].items() for __t in __s))
TOKENS[16]['all'] = list(set(__t for _, __s in TOKENS[16].items() for __t in __s))

# EMBEDDINGS ##################################################################

for __l, __s in TOKENS[1].items():
    # re-encode without token repeats
    __token_x = tf.one_hot(indices=_mmtp._tokenize_scalar(text=''.join(__s), layer_count=N_DEPTH, group_size=4, flatten=True), depth=256, axis=-1)
    # embed
    EMBEDDINGS[1][__l] = MODEL._encoder._encoder.layers[1](MODEL._encoder._encoder.layers[0](__token_x))[:len(__s)]

for __l, __s in TOKENS[4].items():
    # re-encode without token repeats
    __token_x = tf.one_hot(indices=_mmtp._tokenize_scalar(text=''.join(__s), layer_count=N_DEPTH, group_size=4, flatten=True), depth=256, axis=-1)
    # embed
    EMBEDDINGS[4][__l] = MODEL._encoder._encoder.layers[2](MODEL._encoder._encoder.layers[1](MODEL._encoder._encoder.layers[0](__token_x)))[:len(__s)]

for __l, __s in TOKENS[16].items():
    # re-encode without token repeats
    __token_x = tf.one_hot(indices=_mmtp._tokenize_scalar(text=''.join(__s), layer_count=N_DEPTH, group_size=4, flatten=True), depth=256, axis=-1)
    # embed
    EMBEDDINGS[16][__l] = MODEL._encoder(__token_x)[:len(__s)]

# SAVE ########################################################################

_mti.write(data=[__c + ' ' + _mti.label(__c) for __c in TOKENS[1]['all']], path='./metadata.1.tsv', tsv=False)
_mti.write(data=EMBEDDINGS[1]['all'].numpy(), path='./embeddings.1.tsv', tsv=True)

_mti.write(data=[__c + ' ' + _mti.label(__c) for __c in TOKENS[4]['all']], path='./metadata.4.tsv', tsv=False)
_mti.write(data=EMBEDDINGS[4]['all'].numpy(), path='./embeddings.4.tsv', tsv=True)

_mti.write(data=[__c + ' ' + _mti.label(__c) for __c in TOKENS[16]['all']], path='./metadata.16.tsv', tsv=False)
_mti.write(data=EMBEDDINGS[16]['all'].numpy(), path='./embeddings.16.tsv', tsv=True)

# TEST ########################################################################

__s = """class Encoder(tf.keras.models.Model):\n    def __init__(self, depth: int, token_dim: int, encoding_dim: int, embedding_dim: int, latent_dim: int, batch_dim: int=None, attention: bool=False, **kwargs) -> None:\n        super(Encoder, self).__init__(**kwargs)\n        self._encoder = tf.keras.Sequential([\n            tf.keras.Input(shape=(encoding_dim,), batch_size=batch_dim, name='input'), # (B * G ^ D, U)\n            tf.keras.layers.Dense(units=embedding_dim, activation=None, use_bias=False, kernel_initializer='glorot_uniform', bias_initializer=None, name='embed-1'),] # (B * G ^ D, U) => (B * G ^ D, E)\n            + [_mmtl.TokenizeBlock(left_axis=-2, right_axis=-1, token_dim=token_dim, latent_dim=latent_dim, attention=attention, name='tokenize' + (__i + 1) * '-4') for __i in range(depth)]) # (B * G ^ i, E) => (B * G ^ (i-1), E)\n\n    def call(self, x: tf.Tensor) -> tf.Tensor:\n        return self._encoder(x)\n"""

__x = tf.one_hot(indices=_mmtp._tokenize_scalar(text=__s, layer_count=N_DEPTH, group_size=4, flatten=True), depth=256, axis=-1)
__e = MODEL._encoder(__x)
__p = MODEL(__x)
__y = _mmtp.postprocess(__p)

print(__s)
print(__y)
print(_mmtp.compare(__s, __y))
