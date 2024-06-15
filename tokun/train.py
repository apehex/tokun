"""Train tokun from scratch or from a checkpoint."""

import datetime
import functools
import itertools
import math
import os

import tensorflow as tf
import tensorflow_datasets as tfds

import mlable.data
import mlable.optimizers

import tokun.data
import tokun.meta
import tokun.model
import tokun.pipeline

# TOGGLE ######################################################################

IMPORT = False
TRAINING = True
RANDOM = True

# META ########################################################################

N_SEQUENCE_AXIS = 1
N_FEATURE_AXIS = -1

N_TOKEN_DIM = [4, 4, 4] # G, for each block
N_ENCODING_DIM = 256 # U
N_EMBEDDING_DIM = N_ENCODING_DIM # E
N_HIDDEN_DIM = 4 * N_EMBEDDING_DIM # H
N_LATENT_DIM = N_EMBEDDING_DIM # L

N_BATCH_DIM = 128 # number of samples per batch
N_SAMPLE_DIM = 256 # number of characters per sample (=> N_TOKEN_DIM * N_SAMPLE_DIM integers per sample)

N_EPOCHS = 16
N_EPOCHS_RAMPUP = 0
N_EPOCHS_SUSTAIN = 0

R_MIN, R_MAX, R_EXP = tokun.meta.rates(pretrained=IMPORT, normalization=True, base=0.001)

# DERIVED #####################################################################

N_TOKEN_SIZES = list(itertools.accumulate(N_TOKEN_DIM, lambda x, y: x * y)) # in bytes
N_OFFSET_TICKS = [2 ** __i for __i in range(int(math.log(N_TOKEN_SIZES[-1] // 4, 2)))] # in characters

# IMPORT ######################################################################

PATH_IMPORT = os.path.join('models/4x4x4/relu/True/True/3.8.keras')

# LOG #########################################################################

VERSION = tokun.meta.version(units=N_TOKEN_DIM, axis=N_SEQUENCE_AXIS)
DATETIME = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

PATH_LOG = os.path.join('.logs/', *VERSION, DATETIME)
PATH_EXPORT = os.path.join('models/', *VERSION, DATETIME + '.keras')

# DATA ########################################################################

LANG = ['ar', 'de', 'en', 'es', 'hi', 'vi', 'zh']
MLQA_TRAIN = {__l: tfds.load('mlqa/' + __l, split='test', as_supervised=False, shuffle_files=True, data_dir='~/.cache/tensorflow/', batch_size=None) for __l in LANG}
MLQA_TEST = {__l: tfds.load('mlqa/' + __l, split='validation', as_supervised=False, shuffle_files=True, data_dir='~/.cache/tensorflow/', batch_size=None) for __l in LANG}

RANDOM_TRAIN = tokun.data.random_dataset(size=N_BATCH_DIM * 2 ** 10, sample_size=N_SAMPLE_DIM, lower_plane=0, upper_plane=0x40000)
RANDOM_TEST = tokun.data.random_dataset(size=N_BATCH_DIM * 2 ** 8, sample_size=N_SAMPLE_DIM, lower_plane=0, upper_plane=0x40000)

# PREPROCESS MLQA #############################################################

PIPELINE = [
    # offset by 1 to 15 character => (1,) bytes
    *[(functools.partial(tokun.pipeline.offset, ticks=__t), False) for __t in N_OFFSET_TICKS], # (offsets 0, ..., (2 ^ i) - 1) + (offsets 2 ^ i, ..., 2 ^ (i+1) - 1)
    # encode => (4 * S,) int
    (functools.partial(tokun.pipeline.encode, token_size=N_TOKEN_SIZES[-1], sample_size=N_SAMPLE_DIM), True),
    # reshape => (4 * S,) int
    (functools.partial(tf.reshape, shape=(4 * N_SAMPLE_DIM,)), True),
    # one-hot encoding for the targets => (4 * S, E) int (bool)
    ((lambda x: (x, tf.one_hot(x, depth=N_ENCODING_DIM, axis=-1))), True)]

OPERATIONS, REPLACE = zip(*PIPELINE)

MLQA_TRAIN = {__l: mlable.data.process(dataset=__d, feature='context', pipeline=OPERATIONS, replace=REPLACE) for __l, __d in MLQA_TRAIN.items()}
MLQA_TEST = {__l: mlable.data.process(dataset=__d, feature='context', pipeline=OPERATIONS, replace=REPLACE) for __l, __d in MLQA_TEST.items()}

# PREPROCESS RANDOM ###########################################################

PIPELINE = [
    # reshape => (4 * S,) int
    (functools.partial(tf.reshape, shape=(4 * N_SAMPLE_DIM,)), True),
    # one-hot encoding for the targets => (4 * S, E) int (bool)
    ((lambda x: (x, tf.one_hot(x, depth=N_ENCODING_DIM, axis=-1))), True)]

OPERATIONS, REPLACE = zip(*PIPELINE)

RANDOM_TRAIN = mlable.data.process(dataset=RANDOM_TRAIN, feature='', pipeline=OPERATIONS, replace=REPLACE)
RANDOM_TEST = mlable.data.process(dataset=RANDOM_TEST, feature='', pipeline=OPERATIONS, replace=REPLACE)

# COMBINE DATASETS ############################################################

DATASET_TRAIN = RANDOM_TRAIN if RANDOM else MLQA_TRAIN['ar'].concatenate(MLQA_TRAIN['en']).concatenate(MLQA_TRAIN['es']).concatenate(MLQA_TRAIN['de']).concatenate(MLQA_TRAIN['hi']).concatenate(MLQA_TRAIN['vi']).concatenate(MLQA_TRAIN['zh'])
DATASET_TEST = MLQA_TEST['ar'].concatenate(MLQA_TEST['en']).concatenate(MLQA_TEST['es']).concatenate(MLQA_TEST['de']).concatenate(MLQA_TEST['hi']).concatenate(MLQA_TEST['vi']).concatenate(MLQA_TEST['zh'])

# INIT ########################################################################

if IMPORT and os.path.isfile(PATH_IMPORT):
    MODEL = tf.keras.models.load_model(PATH_IMPORT)
else:
    MODEL = tokun.model.AutoEncoder(sequence_axis=N_SEQUENCE_AXIS, feature_axis=N_FEATURE_AXIS,token_dim=N_TOKEN_DIM, encoding_dim=N_ENCODING_DIM, embedding_dim=N_EMBEDDING_DIM, hidden_dim=N_HIDDEN_DIM, latent_dim=N_LATENT_DIM, activation='gelu')

# COMPILE #####################################################################

MODEL.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=R_MAX),
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False, label_smoothing=0., axis=-1, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE, name='loss'),
    metrics=['accuracy'])

# TRAIN #######################################################################

tb_callback = tf.keras.callbacks.TensorBoard(log_dir=PATH_LOG)
cp_callback = tf.keras.callbacks.ModelCheckpoint(PATH_EXPORT, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', save_freq='epoch')
lr_callback = tf.keras.callbacks.LearningRateScheduler(functools.partial(mlable.optimizers.learning_rate_hokusai, lr_min=R_MIN, lr_max=R_MAX, lr_exp=R_EXP, rampup=N_EPOCHS_RAMPUP, sustain=N_EPOCHS_SUSTAIN), verbose=True)

if TRAINING:
    HISTORY = MODEL.fit(
        x=DATASET_TRAIN.batch(N_BATCH_DIM).prefetch(tf.data.AUTOTUNE),
        batch_size=None,
        epochs=N_EPOCHS,
        validation_split=None,
        validation_data=DATASET_TEST.batch(N_BATCH_DIM).prefetch(tf.data.AUTOTUNE),
        validation_freq=list(range(1, N_EPOCHS + 1)),
        verbose=2,
        callbacks=[lr_callback, cp_callback, tb_callback])
