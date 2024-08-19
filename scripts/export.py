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

# TOGGLE ######################################################################

BINARY = True

# MODEL PARAMETERS ############################################################

TOKUN_CONFIG = {
    'token_dim': [4, 4, 4],
    'input_dim': 256,
    'embed_dim': 256,
    'output_dim': 8 if BINARY else 256,
    'sequence_axis': 1}

# META ########################################################################

META_CONFIG = {
    'version': tokun.meta.version(**TOKUN_CONFIG),
    'label': '6.1',}

IO_CONFIG = {
    'model_path': os.path.join('models/', *META_CONFIG['version'], '{}.keras'.format(META_CONFIG['label'])),
    'embed_path': os.path.join('embeddings/', *META_CONFIG['version']),}

# DATA PARAMETERS #############################################################

BATCH_CONFIG = {
    'batch_size': 128,
    'drop_remainder': True,
    'num_parallel_calls': tf.data.AUTOTUNE,}

PIPELINE_CONFIG = {
    'sample_dim': 4 * 512,
    'token_dim': math.prod(TOKUN_CONFIG['token_dim']),
    'offset_ticks': [2 ** __i for __i in range(int(math.log(math.prod(TOKUN_CONFIG['token_dim']) // 4, 2)))]} # in characters

MLQA_CONFIG = {
    'as_supervised': False,
    'shuffle_files': True,
    'batch_size': None,
    'data_dir': '~/.cache/tensorflow/',}

# TRAINING PARAMETERS #########################################################

OPTIMIZER_CONFIG = {
    'learning_rate': 0.0001,
    'weight_decay': 0.1,
    'beta_1': 0.9,
    'beta_2': 0.99,
    'clipnorm': 1.0,}

LOSS_CONFIG = {
    'from_logits': False,
    'label_smoothing': 0.,
    'axis': -1,
    'reduction': 'sum_over_batch_size',
    'name': 'ce_loss',}

# EXPORTING PARAMETERS ########################################################

EXPORT_CONFIG = {
    'token_sizes': list(itertools.accumulate(TOKUN_CONFIG['token_dim'], lambda x, y: x * y)),}

# DATA ########################################################################

LANG = ['ar', 'de', 'en', 'es', 'hi', 'vi', 'zh']

MLQA_TRAIN = {__l: tfds.load('mlqa/' + __l, split='test', **MLQA_CONFIG) for __l in LANG}
MLQA_TEST = {__l: tfds.load('mlqa/' + __l, split='validation', **MLQA_CONFIG) for __l in LANG}

# OUTPUT ENCODING #############################################################

_encode_binary = lambda __x: tf.cast(mlable.ops.expand_base(__x, base=2, depth=TOKUN_CONFIG['output_dim']), dtype=tf.float32)
_encode_categorical = lambda __x: tf.one_hot(__x, depth=TOKUN_CONFIG['output_dim'], axis=-1)
_encode_output = _encode_binary if BINARY else _encode_categorical

# PREPROCESS ##################################################################

PIPELINE = [
    # join the features
    ((lambda x: tf.strings.join(inputs=[x['context'], x['question']], separator='\x1d')), True),
    # offset by 1 to 15 character => (1,) bytes
    *[(functools.partial(tokun.pipeline.offset, ticks=__t), False) for __t in PIPELINE_CONFIG['offset_ticks']], # (offsets 0, ..., (2 ^ i) - 1) + (offsets 2 ^ i, ..., 2 ^ (i+1) - 1)
    # encode => (4 * S,) int
    (functools.partial(tokun.pipeline.encode, token_size=PIPELINE_CONFIG['token_dim'], sample_size=PIPELINE_CONFIG['sample_dim']), True),
    # reshape => (4 * S,) int
    (functools.partial(tf.reshape, shape=(PIPELINE_CONFIG['sample_dim'],)), True),
    # one-hot encoding for the targets => (4 * S, E) int (bool)
    ((lambda __x: (__x, _encode_output(__x))), True)]

OPERATIONS, REPLACE = zip(*PIPELINE)

MLQA_TRAIN = {__l: mlable.data.process(dataset=__d, pipeline=OPERATIONS, replace=REPLACE) for __l, __d in MLQA_TRAIN.items()}
MLQA_TEST = {__l: mlable.data.process(dataset=__d, pipeline=OPERATIONS, replace=REPLACE) for __l, __d in MLQA_TEST.items()}

# METRICS #####################################################################

_Accuracy = mlable.metrics.BinaryGroupAccuracy if BINARY else mlable.metrics.CategoricalGroupAccuracy
_Loss = tf.keras.losses.BinaryCrossentropy if BINARY else tf.keras.losses.CategoricalCrossentropy

# COMPILE ########################################################################

with DISTRIBUTION_STRATEGY.scope():
    # metrics
    byte_accuracy = _Accuracy(group=1, name='byte_accuracy')
    character_accuracy = _Accuracy(group=4, name='character_accuracy')
    token_accuracy = _Accuracy(group=math.prod(TOKUN_CONFIG['token_dim']), name='token_accuracy')
    # weights and config
    MODEL = tf.keras.models.load_model(IO_CONFIG['model_path'], compile=False)
    # compilation
    MODEL.compile(
        optimizer=tf.keras.optimizers.Adam(**OPTIMIZER_CONFIG),
        loss=_Loss(**LOSS_CONFIG),
        metrics=[byte_accuracy, character_accuracy, token_accuracy])

# SAMPLES #####################################################################

SAMPLES = {}
TOKENS = {__i: {} for __i in EXPORT_CONFIG['token_sizes']} # in bytes
EMBEDDINGS = {__i: {} for __i in EXPORT_CONFIG['token_sizes']} # in bytes

for __lang, __dataset in MLQA_TEST.items():
    # compute predictions
    __batch = iter(__dataset.batch(BATCH_CONFIG['batch_size'])) # iterate over batches of samples
    __inputs, __targets = next(__batch)
    __outputs = MODEL(__inputs)
    # sample predictions (inputs, outputs)
    SAMPLES[__lang] = (__targets, __outputs)

# TOKENS ######################################################################

# unique (G ^ i)-tokens
for __lang, __sample in SAMPLES.items():
    for __size in TOKENS:
        # concatenate all the samples in a batch
        __all = tokun.pipeline.postprocess(__sample[0], binary=BINARY, random=False)
        __all = tokun.pipeline.unpack(__all)
        __all = ''.join(__all)
        # save all the unique chunks
        TOKENS[__size][__lang] = tokun.pipeline.chunk(seq=__all, size=__size // 4, repeats=False)

# unique tokens, for all languages
for __size in TOKENS:
    TOKENS[__size]['all'] = list(set(__t for _, __s in TOKENS[__size].items() for __t in __s))

# EMBEDDINGS ##################################################################

for __depth, __size in enumerate(EXPORT_CONFIG['token_sizes']):
    for __lang, __tokens in TOKENS[__size].items():
        # re-encode without token repeats
        __input = tokun.pipeline.preprocess(text=''.join(__tokens), token_size=PIPELINE_CONFIG['token_dim'], expand=[1])
        # UTF-32 embedding
        __embedding = MODEL._encoder._layers[0](__input)
        # iterative CNN tokenization
        for __i in range(__depth + 1):
            __embedding = MODEL._encoder._layers[__i + 1](__embedding)
        # mixed precision: bfloat16 => float32
        __embedding = tf.cast(__embedding, dtype=tf.float32)
        # remove the (tokenized) padding
        EMBEDDINGS[__size][__lang] = tf.squeeze(__embedding)[:len(__tokens)]

# NEIGHBORHOODS ###############################################################

__unit = PIPELINE_CONFIG['token_dim']
__count = 256

TOKENS['local'] = {'all': []}
EMBEDDINGS['local'] = {'all': []}

for __lang, __tokens in TOKENS[__unit].items():
    # stats on the embeddings for the current language
    __std = tf.math.reduce_std(EMBEDDINGS[__unit][__lang], axis=0, keepdims=True)
    __radius = 2. * tf.reduce_mean(__std).numpy()
    # choose a single token
    __t = tokun.pipeline.preprocess(text=random.choice(__tokens), token_size=PIPELINE_CONFIG['token_dim'], expand=[1])
    # encode it
    __e = tf.cast(MODEL._encoder(__t), dtype=tf.float32)
    # add noise to generate random neighbors
    __n = tokun.evaluation.neighbors(point=__e, radius=__radius, count=__count)
    # decode the noisy embeddings
    __d = MODEL._decoder(__n)
    # postprocess
    __r = tokun.pipeline.postprocess(__d, binary=BINARY, random=False)
    __r = ''.join(tokun.pipeline.unpack(__r))
    # chunk
    __m = tokun.pipeline.chunk(seq=__r, size=__unit // 4, repeats=True)
    # save
    TOKENS['local']['all'].extend(__m)
    EMBEDDINGS['local']['all'].append(tf.squeeze(__n))

# merge all the embedding tensors
EMBEDDINGS['local']['all'] = tf.concat(values=EMBEDDINGS['local']['all'], axis=0)

# EXPORT ######################################################################

for __size in TOKENS:
    mlable.data.write(data=[__c + ' ' + mlable.data.label(__c) for __c in TOKENS[__size]['all']][:8192], path=os.path.join(IO_CONFIG['embed_path'], f'./metadata.{__size}.label.tsv'), tsv=False)
    mlable.data.write(data=TOKENS[__size]['all'][:8192], path=os.path.join(IO_CONFIG['embed_path'], f'./metadata.{__size}.tsv'), tsv=False)
    mlable.data.write(data=EMBEDDINGS[__size]['all'].numpy()[:8192], path=os.path.join(IO_CONFIG['embed_path'], f'./embeddings.{__size}.tsv'), tsv=True)
