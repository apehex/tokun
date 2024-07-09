"""Train tokun from scratch or from a checkpoint."""

# SETUP ENV ###################################################################

import os

os.environ['KERAS_BACKEND'] = 'tensorflow'

# LOAD DEPS ###################################################################

import datetime
import functools
import itertools
import math

import keras as ks
import tensorflow as tf
import tensorflow_datasets as tfds

import mlable.data
import mlable.metrics

import tokun.data
import tokun.meta
import tokun.model
import tokun.pipeline

# MIXED PRECISION #############################################################

ks.config.set_dtype_policy("mixed_bfloat16") # mixed_bfloat16 on TPUs

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
TRAINING = True
RANDOM = True

# META ########################################################################

N_SEQUENCE_AXIS = 1
N_FEATURE_AXIS = -1

N_TOKEN_DIM = [4, 16] # G, for each block
N_ENCODING_DIM = 256 # U
N_EMBEDDING_DIM = N_ENCODING_DIM # E

N_BATCH_DIM = 128 # number of samples per batch
N_SAMPLE_DIM = 256 # number of characters per sample (=> N_TOKEN_DIM * N_SAMPLE_DIM integers per sample)

N_EPOCHS = 8

R_0, B_1, B_2 = tokun.meta.rates(pretrained=IMPORT, normalization=True, base=0.001)

CLASS_WEIGHTS = {__c: 0.3 if __c == 0 else 1. for __c in range(N_ENCODING_DIM)} # there are 3 times more 0s than other bytes

# DERIVED #####################################################################

N_TOKEN_SIZES = list(itertools.accumulate(N_TOKEN_DIM, lambda x, y: x * y)) # in bytes
N_OFFSET_TICKS = [2 ** __i for __i in range(int(math.log(N_TOKEN_SIZES[-1] // 4, 2)))] # in characters

# IMPORT ######################################################################

PATH_IMPORT = os.path.join('models/4x16/1/7.3.keras')

# LOG #########################################################################

VERSION = tokun.meta.version(units=N_TOKEN_DIM, axis=N_SEQUENCE_AXIS)
DATETIME = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

PATH_LOG = os.path.join('.logs/', *VERSION, DATETIME)
PATH_EXPORT = os.path.join('models/', *VERSION, DATETIME + '.keras')

# DATA ########################################################################

LANG = ['ar', 'de', 'en', 'es', 'hi', 'vi', 'zh']
MLQA_TRAIN = {__l: tfds.load('mlqa/' + __l, split='test', as_supervised=False, shuffle_files=True, data_dir='~/.cache/tensorflow/', batch_size=None) for __l in LANG}
MLQA_TEST = {__l: tfds.load('mlqa/' + __l, split='validation', as_supervised=False, shuffle_files=True, data_dir='~/.cache/tensorflow/', batch_size=None) for __l in LANG}

RANDOM_TRAIN = tokun.data.random_dataset(size=N_BATCH_DIM * 2 ** 14, sample_size=N_SAMPLE_DIM, lower_plane=0, upper_plane=0x40000)
RANDOM_TEST = tokun.data.random_dataset(size=N_BATCH_DIM * 2 ** 8, sample_size=N_SAMPLE_DIM, lower_plane=0, upper_plane=0x40000)

# PREPROCESS MLQA #############################################################

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
    ((lambda x: (x, tf.one_hot(x, depth=N_ENCODING_DIM, axis=-1))), True)]

OPERATIONS, REPLACE = zip(*PIPELINE)

MLQA_TRAIN = {__l: mlable.data.process(dataset=__d, pipeline=OPERATIONS, replace=REPLACE) for __l, __d in MLQA_TRAIN.items()}
MLQA_TEST = {__l: mlable.data.process(dataset=__d, pipeline=OPERATIONS, replace=REPLACE) for __l, __d in MLQA_TEST.items()}

# PREPROCESS RANDOM ###########################################################

PIPELINE = [
    # reshape => (4 * S,) int
    (functools.partial(tf.reshape, shape=(4 * N_SAMPLE_DIM,)), True),
    # one-hot encoding for the targets => (4 * S, E) int (bool)
    ((lambda x: (x, tf.one_hot(x, depth=N_ENCODING_DIM, axis=-1))), True)]

OPERATIONS, REPLACE = zip(*PIPELINE)

RANDOM_TRAIN = mlable.data.process(dataset=RANDOM_TRAIN, pipeline=OPERATIONS, replace=REPLACE)
RANDOM_TEST = mlable.data.process(dataset=RANDOM_TEST, pipeline=OPERATIONS, replace=REPLACE)

# COMBINE DATASETS ############################################################

DATASET_TRAIN = RANDOM_TRAIN if RANDOM else MLQA_TRAIN['ar'].concatenate(MLQA_TRAIN['en']).concatenate(MLQA_TRAIN['es']).concatenate(MLQA_TRAIN['de']).concatenate(MLQA_TRAIN['hi']).concatenate(MLQA_TRAIN['vi']).concatenate(MLQA_TRAIN['zh'])
DATASET_TEST = MLQA_TEST['ar'].concatenate(MLQA_TEST['en']).concatenate(MLQA_TEST['es']).concatenate(MLQA_TEST['de']).concatenate(MLQA_TEST['hi']).concatenate(MLQA_TEST['vi']).concatenate(MLQA_TEST['zh'])

# INSPECT #####################################################################

print(RANDOM_TRAIN.element_spec)
print(RANDOM_TEST.element_spec)

print(DATASET_TRAIN.element_spec)
print(DATASET_TEST.element_spec)

print('train: {:,}'.format(DATASET_TRAIN.cardinality().numpy()))
print('test:  {:,}'.format(DATASET_TEST.cardinality().numpy()))

# COMPILE ########################################################################

with DISTRIBUTION_STRATEGY.scope():
    # metrics
    byte_accuracy = mlable.metrics.CategoricalGroupAccuracy(group=1, name='byte_accuracy')
    character_accuracy = mlable.metrics.CategoricalGroupAccuracy(group=4, name='character_accuracy')
    token_accuracy = mlable.metrics.CategoricalGroupAccuracy(group=N_TOKEN_SIZES[-1], name='token_accuracy')
    # weights
    MODEL = tokun.model.AutoEncoder(sequence_axis=N_SEQUENCE_AXIS, feature_axis=N_FEATURE_AXIS, token_dim=N_TOKEN_DIM, encoding_dim=N_ENCODING_DIM, embedding_dim=N_EMBEDDING_DIM, activation='gelu')
    if IMPORT and os.path.isfile(PATH_IMPORT): MODEL = ks.models.load_model(PATH_IMPORT, compile=False)
    # compile
    MODEL.compile(
        optimizer=ks.optimizers.Adam(learning_rate=R_0, beta_1=B_1, beta_2=B_2),
        loss=ks.losses.CategoricalCrossentropy(from_logits=False, label_smoothing=0., axis=-1, reduction='sum_over_batch_size', name='loss'),
        metrics=[byte_accuracy, character_accuracy, token_accuracy])

# TRAIN #######################################################################

if TRAINING:
    with DISTRIBUTION_STRATEGY.scope():
        # callbacks
        cp_callback = ks.callbacks.ModelCheckpoint(PATH_EXPORT, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', save_freq='epoch')
        tb_callback = ks.callbacks.TensorBoard(log_dir=PATH_LOG)
        # fit model
        TRAINING_HISTORY = MODEL.fit(
            x=DATASET_TRAIN.batch(N_BATCH_DIM).prefetch(tf.data.AUTOTUNE),
            batch_size=None,
            epochs=N_EPOCHS,
            validation_split=None,
            validation_data=DATASET_TEST.batch(N_BATCH_DIM).prefetch(tf.data.AUTOTUNE),
            validation_freq=list(range(1, N_EPOCHS + 1, 1)),
            class_weight=CLASS_WEIGHTS,
            verbose=1,
            callbacks=[cp_callback, tb_callback])
