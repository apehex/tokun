"""Tensorflow port of the tutorial by Andrej Karpathy: https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/"""

import datetime
import functools
import math
import os

import tensorflow as tf
import tensorflow_datasets as tfds

import mlable.tensorflow.io as _mti
import mlable.tensorflow.optimizers as _mto

import tokun.layers
import tokun.model
import tokun.pipeline

# META ########################################################################

ATTENTION = True
NORMALIZATION = True

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

# LOG #########################################################################

VERSION = 'tokun-' + str(N_TOKEN_DIM ** (N_DEPTH - 1)) + '-keras' + ATTENTION * '-attention' + NORMALIZATION * '-normalization'

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
    *[(functools.partial(tokun.pipeline.offset, ticks=2 ** __i, layer=1, unit=N_TOKEN_DIM), False) for __i in range(2 * (N_DEPTH - 1))], # (offsets 0, ..., (2 ^ i) - 1) + (offsets 2 ^ i, ..., 2 ^ (i+1) - 1)
    # encode => (B * T * S,) int
    (functools.partial(tokun.pipeline.encode, layer_count=N_DEPTH, group_size=N_TOKEN_DIM, sample_size=N_SAMPLE, flatten=True), True),
    # one-hot encoding => (B * T * S, E) int (bool)
    (functools.partial(tf.one_hot, depth=N_ENCODING_DIM, axis=-1), True),
    # replace sample inputs with (inputs, target) for supervised learning
    ((lambda x: (x, x)), True)]

OPERATIONS, REPLACE = zip(*PIPELINE)

TRAIN = {__l: tokun.pipeline.process(dataset=__d, feature='context', pipeline=OPERATIONS, replace=REPLACE) for __l, __d in TRAIN.items()}
TEST = {__l: tokun.pipeline.process(dataset=__d, feature='context', pipeline=OPERATIONS, replace=REPLACE) for __l, __d in TEST.items()}

# INIT ########################################################################

MODEL = tokun.model.AutoEncoder(depth=N_DEPTH, token_dim=N_TOKEN_DIM, encoding_dim=N_ENCODING_DIM, embedding_dim=N_EMBEDDING_DIM, latent_dim=N_LATENT_DIM, batch_dim=None, attention=ATTENTION, normalization=NORMALIZATION)

# compile
MODEL.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=R_MAX),
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False, label_smoothing=0., axis=-1, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE, name='loss'),
    metrics=['accuracy'])

# TRAIN #######################################################################

tb_callback = tf.keras.callbacks.TensorBoard(log_dir=LOGPATH)
lr_callback = tf.keras.callbacks.LearningRateScheduler(functools.partial(_mto.learning_rate_hokusai, lr_min=R_MIN, lr_max=R_MAX, lr_exp=R_EXP, rampup=N_EPOCHS_RAMPUP, sustain=N_EPOCHS_SUSTAIN), verbose=True)

TRAINING_HISTORY = MODEL.fit(
    x=TRAIN['ar'].concatenate(TRAIN['en']).concatenate(TRAIN['es']).concatenate(TRAIN['de']).concatenate(TRAIN['hi']).concatenate(TRAIN['vi']).concatenate(TRAIN['zh']),
    batch_size=N_BATCH,
    epochs=N_EPOCHS,
    validation_split=None,
    validation_data=TEST['zh'], # full of glyphs
    validation_freq=list(range(1, N_EPOCHS + 1, N_EPOCHS // 8)),
    verbose=2,
    callbacks=[lr_callback, tb_callback])

# SAMPLES #####################################################################

SAMPLES = {}
TOKENS = {(N_TOKEN_DIM ** __i): {} for __i in range(N_DEPTH)}
EMBEDDINGS = {(N_TOKEN_DIM ** __i): {} for __i in range(N_DEPTH)}

for __lang in TEST:
    # compute predictions
    __batch = iter(TEST[__lang]) # iterate over batches of samples
    __input = next(__batch)[0] # take input only
    __output = MODEL(__input)
    # sample predictions (inputs, outputs)
    SAMPLES[__lang] = (__input, __output)
    # unique (G ^ i)-tokens
    for __size in TOKENS:
        TOKENS[__size][__lang] = tokun.pipeline.chunk(seq=tokun.pipeline.postprocess(__input), size=__size, repeats=False)

# unique tokens, for all languages
for __size in TOKENS:
    TOKENS[__size]['all'] = list(set(__t for _, __s in TOKENS[__size].items() for __t in __s))

# EMBEDDINGS ##################################################################

for __size in TOKENS:
    for __lang, __tokens in TOKENS[__size].items():
        # embedding depth / nesting
        __depth = int(math.log(__size, N_TOKEN_DIM))
        # re-encode without token repeats
        __input = tf.one_hot(indices=tokun.pipeline._encode_scalar(text=''.join(__tokens), layer_count=N_DEPTH, group_size=N_TOKEN_DIM, flatten=True), depth=N_ENCODING_DIM, axis=-1)
        # UTF-32 embedding
        __embedding = MODEL._encoder._encoder.layers[0](__input)
        # iterative CNN tokenization
        for __i in range(__depth + 1):
            __embedding = MODEL._encoder._encoder.layers[__i + 1](__embedding)
        # remove the (tokenized) padding
        EMBEDDINGS[__size][__lang] = __embedding[:len(__tokens)]

# SAVE ########################################################################

for __size in TOKENS:
    _mti.write(data=[__c + ' ' + _mti.label(__c) for __c in TOKENS[__size]['all']], path='./metadata.' + str(__size) + '.tsv', tsv=False)
    _mti.write(data=EMBEDDINGS[__size]['all'].numpy(), path='./embeddings.' + str(__size) + '.tsv', tsv=True)

# TEST ########################################################################

__s = """class Encoder(tf.keras.models.Model):\n    def __init__(self, depth: int, token_dim: int, encoding_dim: int, embedding_dim: int, latent_dim: int, batch_dim: int=None, attention: bool=False, **kwargs) -> None:\n        super(Encoder, self).__init__(**kwargs)\n        self._encoder = tf.keras.Sequential([\n            tf.keras.Input(shape=(encoding_dim,), batch_size=batch_dim, name='input'), # (B * G ^ D, U)\n            tf.keras.layers.Dense(units=embedding_dim, activation=None, use_bias=False, kernel_initializer='glorot_uniform', bias_initializer=None, name='embed-1'),] # (B * G ^ D, U) => (B * G ^ D, E)\n            + [tokun.layers.TokenizeBlock(left_axis=-2, right_axis=-1, token_dim=token_dim, latent_dim=latent_dim, attention=attention, name='tokenize' + (__i + 1) * '-4') for __i in range(depth)]) # (B * G ^ i, E) => (B * G ^ (i-1), E)\n\n    def call(self, x: tf.Tensor) -> tf.Tensor:\n        return self._encoder(x)\n"""

__x = tf.one_hot(indices=tokun.pipeline._encode_scalar(text=__s, layer_count=N_DEPTH, group_size=N_TOKEN_DIM, flatten=True), depth=N_ENCODING_DIM, axis=-1)
__e = MODEL._encoder(__x)
__p = MODEL(__x)
__y = tokun.pipeline.postprocess(__p)

print(__s)
print(__y)
print(tokun.pipeline.compare(__s, __y))
