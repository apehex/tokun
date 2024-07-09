"""Compute tokens and embeddings on MLQA."""

# SETUP ENV ###################################################################

import os

os.environ['KERAS_BACKEND'] = 'tensorflow'

# LOAD DEPS ###################################################################

import datetime
import functools
import itertools
import math
import random

import keras as ks
import tensorflow as tf
import tensorflow_datasets as tfds

import mlable.data
import mlable.metrics

import tokun.evaluation
import tokun.meta
import tokun.model
import tokun.pipeline

# DEVICES #####################################################################

tf.debugging.set_log_device_placement(False)

CPU = tf.config.list_logical_devices('CPU')
GPU = tf.config.list_logical_devices('GPU')
TPU = tf.config.list_logical_devices('TPU')

if TPU:
    RESOLVER = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(RESOLVER)
    tf.tpu.experimental.initialize_tpu_system(RESOLVER)
    DISTRIBUTION_STRATEGY = tf.distribute.TPUStrategy(RESOLVER)
elif GPU:
    DISTRIBUTION_STRATEGY = tf.distribute.MirroredStrategy(GPU)
else:
    DISTRIBUTION_STRATEGY = tf.distribute.MirroredStrategy(CPU)

print(DISTRIBUTION_STRATEGY)

# META ########################################################################

N_SEQUENCE_AXIS = 1
N_TOKEN_DIM = [4, 16] # G, for each block

N_BATCH_DIM = 128 # number of samples per batch
N_SAMPLE_DIM = 128 # number of characters per sample (=> N_TOKEN_DIM * N_SAMPLE_DIM integers per sample)

# DERIVED #####################################################################

N_TOKEN_SIZES = list(itertools.accumulate(N_TOKEN_DIM, lambda x, y: x * y)) # in bytes
N_OFFSET_TICKS = [2 ** __i for __i in range(int(math.log(N_TOKEN_SIZES[-1] // 4, 2)))] # in characters

# IMPORT MODEL ################################################################

VERSION = tokun.meta.version(units=N_TOKEN_DIM, axis=N_SEQUENCE_AXIS)
LABEL = '7.3'

PATH_IMPORT = os.path.join('models/', *VERSION, '{}.keras'.format(LABEL))

# INIT MODEL ##################################################################

with DISTRIBUTION_STRATEGY.scope():
    # metrics
    byte_accuracy = mlable.metrics.CategoricalGroupAccuracy(group=1, name='byte_accuracy')
    character_accuracy = mlable.metrics.CategoricalGroupAccuracy(group=4, name='character_accuracy')
    token_accuracy = mlable.metrics.CategoricalGroupAccuracy(group=N_TOKEN_SIZES[-1], name='token_accuracy')
    # weights and config
    MODEL = ks.saving.load_model(PATH_IMPORT, compile=False)
    # compilation
    MODEL.compile(
        optimizer=ks.optimizers.Adam(learning_rate=0.0001),
        loss=ks.losses.CategoricalCrossentropy(from_logits=False, label_smoothing=0., axis=-1, reduction='sum_over_batch_size', name='cce_loss'),
        metrics=[byte_accuracy, character_accuracy, token_accuracy])

# DATA ########################################################################

LANG = ['ar', 'de', 'en', 'es', 'hi', 'vi', 'zh']
MLQA_TRAIN = {__l: tfds.load('mlqa/' + __l, split='test', as_supervised=False, shuffle_files=True, data_dir='~/.cache/tensorflow/', batch_size=None) for __l in LANG}
MLQA_TEST = {__l: tfds.load('mlqa/' + __l, split='validation', as_supervised=False, shuffle_files=True, data_dir='~/.cache/tensorflow/', batch_size=None) for __l in LANG}

# PREPROCESS ##################################################################

PIPELINE = [
    # join the features
    ((lambda x: tf.strings.join(inputs=[x['context'], x['question'], x['answers']['text']], separator='\x1d')), True),
    # offset by 1 to 15 character => (1,) bytes
    *[(functools.partial(tokun.pipeline.offset, ticks=__t), False) for __t in N_OFFSET_TICKS], # (offsets 0, ..., (2 ^ i) - 1) + (offsets 2 ^ i, ..., 2 ^ (i+1) - 1)
    # encode => (4 * S,) int
    (functools.partial(tokun.pipeline.encode, token_size=N_TOKEN_SIZES[-1], sample_size=N_SAMPLE_DIM), True),
    # reshape => (4 * S,) int
    (functools.partial(tf.reshape, shape=(4 * N_SAMPLE_DIM,)), True),
    # one-hot encoding for the targets => (4 * S, E) int (bool)
    ((lambda x: (x, tf.one_hot(x, depth=256, axis=-1))), True)]

OPERATIONS, REPLACE = zip(*PIPELINE)

MLQA_TRAIN = {__l: mlable.data.process(dataset=__d, pipeline=OPERATIONS, replace=REPLACE) for __l, __d in MLQA_TRAIN.items()}
MLQA_TEST = {__l: mlable.data.process(dataset=__d, pipeline=OPERATIONS, replace=REPLACE) for __l, __d in MLQA_TEST.items()}

# SAMPLES #####################################################################

SAMPLES = {}
TOKENS = {__i: {} for __i in N_TOKEN_SIZES} # in bytes
EMBEDDINGS = {__i: {} for __i in N_TOKEN_SIZES} # in bytes

for __lang, __dataset in MLQA_TEST.items():
    # compute predictions
    __batch = iter(__dataset.batch(N_BATCH_DIM)) # iterate over batches of samples
    __inputs, __targets = next(__batch)
    __outputs = MODEL(__inputs)
    # sample predictions (inputs, outputs)
    SAMPLES[__lang] = (__targets, __outputs)

# TOKENS ######################################################################

# unique (G ^ i)-tokens
for __lang, __sample in SAMPLES.items():
    for __size in TOKENS:
        TOKENS[__size][__lang] = tokun.pipeline.chunk(seq=tokun.pipeline.postprocess(__sample[0]), size=__size // 4, repeats=False)

# unique tokens, for all languages
for __size in TOKENS:
    TOKENS[__size]['all'] = list(set(__t for _, __s in TOKENS[__size].items() for __t in __s))

# EMBEDDINGS ##################################################################

for __depth, __size in enumerate(N_TOKEN_SIZES):
    for __lang, __tokens in TOKENS[__size].items():
        # re-encode without token repeats
        __input = tokun.pipeline.preprocess(text=''.join(__tokens), token_size=math.prod(N_TOKEN_DIM), expand=N_SEQUENCE_AXIS * [1])
        # UTF-32 embedding
        __embedding = MODEL._encoder._encoder.layers[0](__input)
        # iterative CNN tokenization
        for __i in range(__depth + 1):
            __embedding = MODEL._encoder._encoder.layers[__i + 1](__embedding)
        # remove the (tokenized) padding
        EMBEDDINGS[__size][__lang] = ks.ops.squeeze(__embedding)[:len(__tokens)]

# NEIGHBORHOODS ###############################################################

__unit = N_TOKEN_SIZES[-1]
__count = 256

TOKENS['local'] = {'all': []}
EMBEDDINGS['local'] = {'all': []}

for __lang, __tokens in TOKENS[__unit].items():
    # stats on the embeddings for the current language
    __std = ks.ops.std(EMBEDDINGS[__unit][__lang], axis=1, keepdims=True)
    __radius = 2. ** (1 - math.log(__unit, 4)) * ks.ops.mean(__std).numpy()
    # choose a single token
    __t = tokun.pipeline.preprocess(text=random.choice(__tokens), token_size=math.prod(N_TOKEN_DIM), expand=N_SEQUENCE_AXIS * [1])
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
    EMBEDDINGS['local']['all'].append(ks.ops.squeeze(__n))

# merge all the embedding tensors
EMBEDDINGS['local']['all'] = ks.ops.concatenate(xs=EMBEDDINGS['local']['all'], axis=0)

# EXPORT ######################################################################

for __size in TOKENS:
    mlable.data.write(data=[__c + ' ' + mlable.data.label(__c) for __c in TOKENS[__size]['all']], path='./metadata.' + str(__size) + '.label.tsv', tsv=False)
    mlable.data.write(data=TOKENS[__size]['all'], path='./metadata.' + str(__size) + '.tsv', tsv=False)
    mlable.data.write(data=EMBEDDINGS[__size]['all'].numpy(), path='./embeddings.' + str(__size) + '.tsv', tsv=True)
