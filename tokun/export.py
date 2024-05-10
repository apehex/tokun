"""Tensorflow port of the tutorial by Andrej Karpathy: https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/"""

import datetime
import functools
import math
import os

import tensorflow as tf
import tensorflow_datasets as tfds

import mlable.tensorflow.io as _mti

import tokun.meta
import tokun.model
import tokun.pipeline

# META ########################################################################

ATTENTION = True
NORMALIZATION = True

N_TOKEN_DIM = [4, 4, 4] # G, for each block

N_BATCH = 128 # number of samples per batch
N_SAMPLE = 128 # number of characters per sample (=> N_TOKEN_DIM * N_SAMPLE integers per sample)

# DERIVED #####################################################################

TOKEN_LENGTH = math.prod(N_TOKEN_DIM) # in bytes
OFFSET_TICKS = [2 ** __i for __i in range(math.log(TOKEN_LENGTH // 4, 2))] # in characters

# LOG #########################################################################

VERSION = tokun.meta.version(groups=N_TOKEN_DIM, attention=ATTENTION, normalization=NORMALIZATION)
DATETIME = '20240509-211600'

PATH_MODEL = os.path.join('models/', *VERSION, DATETIME + '.keras')

# DATA ########################################################################

LANG = ['ar', 'de', 'en', 'es', 'hi', 'vi', 'zh']
TRAIN = {__l: tfds.load('mlqa/' + __l, split='test', as_supervised=False, shuffle_files=True, data_dir='~/.cache/tensorflow/', batch_size=N_BATCH) for __l in LANG}
TEST = {__l: tfds.load('mlqa/' + __l, split='validation', as_supervised=False, shuffle_files=True, data_dir='~/.cache/tensorflow/', batch_size=N_BATCH) for __l in LANG}

# PREPROCESS ##################################################################

PIPELINE = [
    # offset by 1 to 15 character => (B, 1) bytes
    *[(functools.partial(tokun.pipeline.offset, ticks=__t), False) for __t in OFFSET_TICKS], # (offsets 0, ..., (2 ^ i) - 1) + (offsets 2 ^ i, ..., 2 ^ (i+1) - 1)
    # encode => (B, G * S,) int
    (functools.partial(tokun.pipeline.encode, token_size=TOKEN_LENGTH, sample_size=N_SAMPLE, flatten=True), True),
    # reshape => (B * G * S,) int
    (functools.partial(tokun.pipeline.reshape, groups=N_TOKEN_DIM, flatten=True), True),
    # one-hot encoding => (B * G * S, E) int (bool)
    (functools.partial(tf.one_hot, depth=N_ENCODING_DIM, axis=-1), True),
    # replace sample inputs with (input, target) for supervised learning
    ((lambda x: (x, x)), True)]

OPERATIONS, REPLACE = zip(*PIPELINE)

TRAIN = {__l: tokun.pipeline.process(dataset=__d, feature='context', pipeline=OPERATIONS, replace=REPLACE) for __l, __d in TRAIN.items()}
TEST = {__l: tokun.pipeline.process(dataset=__d, feature='context', pipeline=OPERATIONS, replace=REPLACE) for __l, __d in TEST.items()}

# LOAD ########################################################################

MODEL = keras.models.load_model(PATH_MODEL)

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

# TOKENS ######################################################################

# unique (G ^ i)-tokens
for __lang, __sample in SAMPLES.items():
    for __size in TOKENS:
        TOKENS[__size][__lang] = tokun.pipeline.chunk(sequence=tokun.pipeline.postprocess(__sample[0]), size=__size, repeats=False)

# unique tokens, for all languages
for __size in TOKENS:
    TOKENS[__size]['all'] = list(set(__t for _, __s in TOKENS[__size].items() for __t in __s))

# EMBEDDINGS ##################################################################

for __size in TOKENS:
    for __lang, __tokens in TOKENS[__size].items():
        # embedding depth / nesting
        __depth = int(math.log(__size, N_TOKEN_DIM))
        # re-encode without token repeats
        __input = tf.one_hot(indices=tokun.pipeline.encode(data=''.join(__tokens), groups=N_TOKEN_DIM, flatten=True), depth=N_ENCODING_DIM, axis=-1)
        # UTF-32 embedding
        __embedding = MODEL._encoder._encoder.layers[0](__input)
        # iterative CNN tokenization
        for __i in range(__depth + 1):
            __embedding = MODEL._encoder._encoder.layers[__i + 1](__embedding)
        # remove the (tokenized) padding
        EMBEDDINGS[__size][__lang] = __embedding[:len(__tokens)]

# EXPORT ######################################################################

for __size in TOKENS:
    _mti.write(data=[__c + ' ' + _mti.label(__c) for __c in TOKENS[__size]['all']], path='./metadata.' + str(__size) + '.label.tsv', tsv=False)
    _mti.write(data=TOKENS[__size]['all'], path='./metadata.' + str(__size) + '.tsv', tsv=False)
    _mti.write(data=EMBEDDINGS[__size]['all'].numpy(), path='./embeddings.' + str(__size) + '.tsv', tsv=True)
