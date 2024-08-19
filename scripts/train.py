"""Train tokun from scratch or from a checkpoint."""

import datetime
import functools
import itertools
import math
import os
import urllib

import tensorflow as tf
import tensorflow_datasets as tfds

import mlable.data
import mlable.metrics

import tokun.data
import tokun.meta
import tokun.model
import tokun.pipeline

# MIXED PRECISION #############################################################

tf.keras.mixed_precision.set_global_policy('mixed_float16') # mixed_bfloat16 on TPUs

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

IMPORT = False
DOWNLOAD = False
TRAINING = True
RANDOM = True
BINARY = True

# MODEL PARAMETERS ############################################################

TOKUN_CONFIG = {
    'sequence_axis': 1,
    'feature_axis': -1,
    'token_dim': [4, 4, 4], # G, for each block
    'input_dim': 256, # U_i (bytes)
    'output_dim': 8 if BINARY else 256, # U_o (8 bits)
    'embedding_dim': 256, # E
    'output': 'binary' if BINARY else 'categorical',
    'activation': 'sigmoid',}

# DERIVED MODEL PARAMETERS ####################################################

VERSION_CONFIG = {
    'token_dim': TOKUN_CONFIG['token_dim'],
    'input_dim': TOKUN_CONFIG['input_dim'],
    'embed_dim': TOKUN_CONFIG['embedding_dim'],
    'output_dim': TOKUN_CONFIG['output_dim'],
    'sequence_axis': TOKUN_CONFIG['sequence_axis']}

META_CONFIG = {
    'version': tokun.meta.version(**VERSION_CONFIG),
    'label': '6.1',}

IO_CONFIG = {
    'url': 'https://github.com/apehex/tokun/raw/main/models/{}/{}/{}/{}.keras'.format(*META_CONFIG['version'], META_CONFIG['label']),
    'path': 'tokun.keras',}

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

RANDOM_CONFIG = {
    'sample_size': PIPELINE_CONFIG['sample_dim'] // 4, # in characters
    'lower_plane': 0,
    'upper_plane': 0x40000,
    'binary': False,}

# TRAINING PARAMETERS #########################################################

OPTIMIZER_CONFIG = {
    'learning_rate': 0.001 * (0.1 if IMPORT else 1.0) * 0.1,
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

CHECKPOINT_CONFIG = {
    'filepath': IO_CONFIG['path'],
    'monitor': 'val_loss',
    'mode': 'auto',
    'save_freq': 'epoch',
    'save_best_only': False,
    'save_weights_only': False,
    'verbose': 1,}

TENSORBOARD_CONFIG = {
    'log_dir': os.path.join('.logs/', *META_CONFIG['version'], datetime.datetime.now().strftime("%Y%m%d-%H%M%S")),
    'histogram_freq': 1,
    'embeddings_freq': 0,
    'profile_batch': (128, 256),
    'write_graph': False,
    'write_images': True,}

TRAINING_CONFIG = {
    'epochs': 8,
    'batch_size': None,
    'validation_split': None,
    'validation_freq': list(range(1, 9)),
    'class_weight': {__c: 0.3 if __c == 0 else 1. for __c in range(TOKUN_CONFIG['input_dim'])}, # there are 3 times more 0s than other bytes
    'verbose': 1,}

# IMPORT ######################################################################

if IMPORT and DOWNLOAD:
    urllib.request.urlretrieve(IO_CONFIG['url'], IO_CONFIG['path'])

# RANDOM DATASET ##############################################################

RANDOM_TRAIN = tokun.data.random_dataset(size=BATCH_CONFIG['batch_size'] * 2**14, **RANDOM_CONFIG)
RANDOM_TEST = tokun.data.random_dataset(size=BATCH_CONFIG['batch_size'] * 2**8, **RANDOM_CONFIG)

# MLQA DATASET ################################################################

MLQA_LANGUAGES = ['ar', 'de', 'en', 'es', 'hi', 'vi', 'zh']

MLQA_TRAIN = {__l: tfds.load('mlqa/' + __l, split='test', **MLQA_CONFIG) for __l in MLQA_LANGUAGES}
MLQA_TEST = {__l: tfds.load('mlqa/' + __l, split='validation', **MLQA_CONFIG) for __l in MLQA_LANGUAGES}

# OUTPUT ENCODING #############################################################

_encode_binary = lambda __x: tf.cast(mlable.ops.expand_base(__x, depth=TOKUN_CONFIG['output_dim'], base=2), dtype=tf.float32)
_encode_categorical = lambda __x: tf.one_hot(__x, depth=TOKUN_CONFIG['output_dim'], axis=-1)
_encode_output = _encode_binary if BINARY else _encode_categorical

# PREPROCESS MLQA #############################################################

PIPELINE = [
    # join the features
    ((lambda __x: tf.strings.join(inputs=[__x['context'], __x['question']], separator='\u001d')), True),
    # offset by 1 to 15 character => (B,) scalar bytes
    *[(functools.partial(tokun.pipeline.offset, ticks=__t), False) for __t in PIPELINE_CONFIG['offset_ticks']], # (offsets 0, ..., (2 ^ i) - 1) + (offsets 2 ^ i, ..., 2 ^ (i+1) - 1)
    # encode => (B, 4 * S,) int (4 UTF-32 bytes per character)
    (functools.partial(tokun.pipeline.encode, token_size=PIPELINE_CONFIG['token_dim'], sample_size=PIPELINE_CONFIG['sample_dim'], dtype=tf.int32), True),
    # reshape => (B, 4 * S,) int
    (functools.partial(tf.reshape, shape=(BATCH_CONFIG['batch_size'], PIPELINE_CONFIG['sample_dim'],)), True),
    # encode classes on 8 bits for the 256 possibilities / byte
    ((lambda __x: (__x, _encode_output(__x))), True)]

OPERATIONS, REPLACE = zip(*PIPELINE)

MLQA_TRAIN = {__l: mlable.data.process(dataset=__d.batch(**BATCH_CONFIG), pipeline=OPERATIONS, replace=REPLACE) for __l, __d in MLQA_TRAIN.items()}
MLQA_TEST = {__l: mlable.data.process(dataset=__d.batch(**BATCH_CONFIG), pipeline=OPERATIONS, replace=REPLACE) for __l, __d in MLQA_TEST.items()}

# PREPROCESS RANDOM ###########################################################

PIPELINE = [
    # reshape each sample => (B, 4 * S,) int
    (functools.partial(tf.reshape, shape=(BATCH_CONFIG['batch_size'], PIPELINE_CONFIG['sample_dim'],)), True),
    # encode classes on 8 bits for the 256 possibilities / byte
    ((lambda __x: (__x, _encode_output(__x))), True)]

OPERATIONS, REPLACE = zip(*PIPELINE)

RANDOM_TRAIN = mlable.data.process(dataset=RANDOM_TRAIN.batch(**BATCH_CONFIG), pipeline=OPERATIONS, replace=REPLACE)
RANDOM_TEST = mlable.data.process(dataset=RANDOM_TEST.batch(**BATCH_CONFIG), pipeline=OPERATIONS, replace=REPLACE)

# COMBINE DATASETS ############################################################

MLQA_TRAIN_ALL = functools.reduce(lambda __l, __r: __l.concatenate(__r), MLQA_TRAIN.values())
MLQA_TEST_ALL = functools.reduce(lambda __l, __r: __l.concatenate(__r), MLQA_TEST.values())

DATASET_TRAIN = RANDOM_TRAIN if RANDOM else MLQA_TRAIN_ALL
DATASET_TEST = MLQA_TEST_ALL

# INSPECT #####################################################################

print(RANDOM_TRAIN.element_spec)
print(RANDOM_TEST.element_spec)

print(DATASET_TRAIN.element_spec)
print(DATASET_TEST.element_spec)

print('train: {:,}'.format(DATASET_TRAIN.cardinality().numpy()))
print('test:  {:,}'.format(DATASET_TEST.cardinality().numpy()))

# METRICS #####################################################################

_Accuracy = mlable.metrics.BinaryGroupAccuracy if BINARY else mlable.metrics.CategoricalGroupAccuracy
_Loss = tf.keras.losses.BinaryCrossentropy if BINARY else tf.keras.losses.CategoricalCrossentropy

# COMPILE ########################################################################

with DISTRIBUTION_STRATEGY.scope():
    # metrics
    byte_accuracy = _Accuracy(group=1, name='byte_accuracy')
    character_accuracy = _Accuracy(group=4, name='character_accuracy')
    token_accuracy = _Accuracy(group=PIPELINE_CONFIG['token_dim'], name='token_accuracy')
    # weights
    MODEL = tokun.model.AutoEncoder(**TOKUN_CONFIG)
    if IMPORT and os.path.isfile(IO_CONFIG['path']): MODEL = tf.keras.models.load_model(IO_CONFIG['path'], compile=False)
    # compile
    MODEL.compile(
        optimizer=tf.keras.optimizers.AdamW(**OPTIMIZER_CONFIG),
        loss=_Loss(**LOSS_CONFIG),
        weighted_metrics=[byte_accuracy, character_accuracy, token_accuracy])

# TRAIN #######################################################################

if TRAINING:
    with DISTRIBUTION_STRATEGY.scope():
        # callbacks
        cp_callback = tf.keras.callbacks.ModelCheckpoint(**CHECKPOINT_CONFIG)
        tb_callback = tf.keras.callbacks.TensorBoard(**TENSORBOARD_CONFIG)
        # fit model
        TRAINING_HISTORY = MODEL.fit(
            x=DATASET_TRAIN.prefetch(tf.data.AUTOTUNE),
            validation_data=DATASET_TEST.prefetch(tf.data.AUTOTUNE),
            callbacks=[cp_callback, tb_callback],
            **TRAINING_CONFIG)
