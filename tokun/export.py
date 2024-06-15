"""Compute tokens and embeddings on MLQA."""

import datetime
import functools
import itertools
import math
import os
import random

import tensorflow as tf
import tensorflow_datasets as tfds

import mlable.data
import mlable.io

import tokun.evaluation
import tokun.meta
import tokun.model
import tokun.pipeline

# META ########################################################################

N_SEQUENCE_AXIS = 1
N_TOKEN_DIM = [4, 4] # G, for each block

N_BATCH_DIM = 128 # number of samples per batch
N_SAMPLE_DIM = 128 # number of characters per sample (=> N_TOKEN_DIM * N_SAMPLE_DIM integers per sample)

# DERIVED #####################################################################

N_TOKEN_SIZES = list(itertools.accumulate(N_TOKEN_DIM, lambda x, y: x * y)) # in bytes
N_OFFSET_TICKS = [2 ** __i for __i in range(int(math.log(N_TOKEN_SIZES[-1] // 4, 2)))] # in characters

# IMPORT ######################################################################

VERSION = tokun.meta.version(units=N_TOKEN_DIM, axis=N_SEQUENCE_AXIS)
LABEL = '8.5'

PATH_IMPORT = os.path.join('models/', *VERSION, '{}.keras'.format(LABEL))

MODEL = keras.models.load_model(PATH_IMPORT)

# DATA ########################################################################

LANG = ['ar', 'de', 'en', 'es', 'hi', 'vi', 'zh']
MLQA_TRAIN = {__l: tfds.load('mlqa/' + __l, split='test', as_supervised=False, shuffle_files=True, data_dir='~/.cache/tensorflow/', batch_size=None) for __l in LANG}
MLQA_TEST = {__l: tfds.load('mlqa/' + __l, split='validation', as_supervised=False, shuffle_files=True, data_dir='~/.cache/tensorflow/', batch_size=None) for __l in LANG}

# PREPROCESS ##################################################################

PIPELINE = [
    # offset by 1 to 15 character => (B, 1) bytes
    *[(functools.partial(tokun.pipeline.offset, ticks=__t), False) for __t in N_OFFSET_TICKS], # (offsets 0, ..., (2 ^ i) - 1) + (offsets 2 ^ i, ..., 2 ^ (i+1) - 1)
    # encode => (B, G * S,) int
    (functools.partial(tokun.pipeline.encode, token_size=N_TOKEN_SIZES[-1], sample_size=N_SAMPLE_DIM), True),
    # reshape => (B * G * S,) int
    (functools.partial(tf.reshape, shape=(4 * N_SAMPLE_DIM,)), True),
    # one-hot encoding for the targets => (4 * S, E) int (bool)
    ((lambda x: (x, tf.one_hot(x, depth=N_ENCODING_DIM, axis=-1))), True)]

OPERATIONS, REPLACE = zip(*PIPELINE)

MLQA_TRAIN = {__l: mlable.data.process(dataset=__d, feature='context', pipeline=OPERATIONS, replace=REPLACE) for __l, __d in MLQA_TRAIN.items()}
MLQA_TEST = {__l: mlable.data.process(dataset=__d, feature='context', pipeline=OPERATIONS, replace=REPLACE) for __l, __d in MLQA_TEST.items()}

# SAMPLES #####################################################################

SAMPLES = {}
TOKENS = {__i: {} for __i in N_TOKEN_SIZES} # in bytes
EMBEDDINGS = {__i: {} for __i in N_TOKEN_SIZES} # in bytes

for __lang, __dataset in MLQA_TEST.items():
    # compute predictions
    __batch = iter(__dataset.batch(N_BATCH_DIM)) # iterate over batches of samples
    __input = next(__batch)[0] # take input only
    __output = MODEL(__input)
    # sample predictions (inputs, outputs)
    SAMPLES[__lang] = (__input, __output)

# TOKENS ######################################################################

# unique (G ^ i)-tokens
for __lang, __sample in SAMPLES.items():
    for __size in TOKENS:
        TOKENS[__size][__lang] = tokun.pipeline.chunk(sequence=tokun.pipeline.postprocess(__sample[0]), size=__size // 4, repeats=False)

# unique tokens, for all languages
for __size in TOKENS:
    TOKENS[__size]['all'] = list(set(__t for _, __s in TOKENS[__size].items() for __t in __s))

# EMBEDDINGS ##################################################################

for __depth, __size in enumerate(N_TOKEN_SIZES):
    for __lang, __tokens in TOKENS[__size].items():
        # re-encode without token repeats
        __input = tokun.pipeline.preprocess(text=''.join(__tokens), groups=N_TOKEN_DIM, expand=N_SEQUENCE_AXIS * [1], flatten=True)
        # UTF-32 embedding
        __embedding = MODEL._encoder._encoder.layers[0](__input)
        # iterative CNN tokenization
        for __i in range(__depth + 1):
            __embedding = MODEL._encoder._encoder.layers[__i + 1](__embedding)
        # remove the (tokenized) padding
        EMBEDDINGS[__size][__lang] = tf.squeeze(__embedding)[:len(__tokens)]

# NEIGHBORHOODS ###############################################################

__unit = N_TOKEN_SIZES[-1]
__count = 256

TOKENS['local'] = {'all': []}
EMBEDDINGS['local'] = {'all': []}

for __lang, __tokens in TOKENS[__unit].items():
    # stats on the embeddings for the current language
    __std = tf.math.reduce_std(EMBEDDINGS[__unit][__lang], axis=0, keepdims=True)
    __radius = 2. * tf.reduce_mean(__std).numpy()
    # choose a single token
    __t = tokun.pipeline.preprocess(text=random.choice(__tokens), groups=N_TOKEN_DIM, expand=N_SEQUENCE_AXIS * [1], flatten=True)
    # encode it
    __e = MODEL._encoder(__t)
    # add noise to generate random neighbors
    __n = tokun.evaluation.neighbors(point=__e, radius=__radius, count=__count)
    # decode the noisy embeddings
    __d = MODEL._decoder(__n)
    # postprocess
    __m = tokun.pipeline.chunk(seq=tokun.pipeline.postprocess(__d), size=__unit // 4, repeats=True)
    # save
    TOKENS['local']['all'].extend(__m)
    EMBEDDINGS['local']['all'].append(tf.squeeze(__n))

# merge all the embedding tensors
EMBEDDINGS['local']['all'] = tf.concat(values=EMBEDDINGS['local']['all'], axis=0)

# EXPORT ######################################################################

for __size in TOKENS:
    mlable.io.write(data=[__c + ' ' + mlable.io.label(__c) for __c in TOKENS[__size]['all']], path='./metadata.' + str(__size) + '.label.tsv', tsv=False)
    mlable.io.write(data=TOKENS[__size]['all'], path='./metadata.' + str(__size) + '.tsv', tsv=False)
    mlable.io.write(data=EMBEDDINGS[__size]['all'].numpy(), path='./embeddings.' + str(__size) + '.tsv', tsv=True)
