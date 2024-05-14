"""Train tokun from scratch or from a checkpoint."""

import datetime
import functools
import itertools
import math
import os

import tensorflow as tf
import tensorflow_datasets as tfds

import mlable.tensorflow.optimizers as _mto

import tokun.meta
import tokun.model
import tokun.pipeline

# TOGGLE ######################################################################

IMPORT = True
TRAINING = True

# META ########################################################################

ATTENTION = True
NORMALIZATION = True

N_TOKEN_DIM = [4, 4, 4] # G, for each block
N_ENCODING_DIM = 256 # U
N_EMBEDDING_DIM = N_ENCODING_DIM # E
N_LATENT_DIM = N_EMBEDDING_DIM # L

N_EPOCHS = 8
N_EPOCHS_RAMPUP = 0
N_EPOCHS_SUSTAIN = 0

N_BATCH = 128 # number of samples per batch
N_SAMPLE = 256 # number of characters per sample (=> N_TOKEN_DIM * N_SAMPLE integers per sample)

R_MIN, R_MAX, R_EXP = tokun.meta.rates(pretrained=IMPORT, normalization=NORMALIZATION, base=0.001)

# DERIVED #####################################################################

TOKEN_SIZES = list(itertools.accumulate(N_TOKEN_DIM, lambda x, y: x * y)) # in bytes
OFFSET_TICKS = [2 ** __i for __i in range(int(math.log(TOKEN_SIZES[-1] // 4, 2)))] # in characters

# IMPORT ######################################################################

PATH_IMPORT = os.path.join('models/4x4x4/True/True/0.99996.keras')

# LOG #########################################################################

VERSION = tokun.meta.version(groups=N_TOKEN_DIM, attention=ATTENTION, normalization=NORMALIZATION)
DATETIME = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

PATH_LOG = os.path.join('.logs/', *VERSION, DATETIME)
PATH_EXPORT = os.path.join('models/', *VERSION, DATETIME + '.keras')

# DATA ########################################################################

LANG = ['ar', 'de', 'en', 'es', 'hi', 'vi', 'zh']
TRAIN = {__l: tfds.load('mlqa/' + __l, split='test', as_supervised=False, shuffle_files=True, data_dir='~/.cache/tensorflow/', batch_size=N_BATCH) for __l in LANG}
TEST = {__l: tfds.load('mlqa/' + __l, split='validation', as_supervised=False, shuffle_files=True, data_dir='~/.cache/tensorflow/', batch_size=N_BATCH) for __l in LANG}

# PREPROCESS ##################################################################

PIPELINE = [
    # offset by 1 to 15 character => (B, 1) bytes
    *[(functools.partial(tokun.pipeline.offset, ticks=__t), False) for __t in OFFSET_TICKS], # (offsets 0, ..., (2 ^ i) - 1) + (offsets 2 ^ i, ..., 2 ^ (i+1) - 1)
    # encode => (B, G * S,) int
    (functools.partial(tokun.pipeline.encode, token_size=TOKEN_SIZES[-1], sample_size=N_SAMPLE), True),
    # reshape => (B * G * S,) int
    (functools.partial(tokun.pipeline.reshape, groups=N_TOKEN_DIM, flatten=True), True),
    # one-hot encoding => (B * G * S, E) int (bool)
    (functools.partial(tf.one_hot, depth=N_ENCODING_DIM, axis=-1), True),
    # replace sample inputs with (input, target) for supervised learning
    ((lambda x: (x, x)), True)]

OPERATIONS, REPLACE = zip(*PIPELINE)

TRAIN = {__l: tokun.pipeline.process(dataset=__d, feature='context', pipeline=OPERATIONS, replace=REPLACE) for __l, __d in TRAIN.items()}
TEST = {__l: tokun.pipeline.process(dataset=__d, feature='context', pipeline=OPERATIONS, replace=REPLACE) for __l, __d in TEST.items()}

# INIT ########################################################################

if IMPORT and os.path.isfile(PATH_IMPORT):
    MODEL = tf.keras.models.load_model(PATH_IMPORT)
else:
    MODEL = tokun.model.AutoEncoder(token_dim=N_TOKEN_DIM, encoding_dim=N_ENCODING_DIM, embedding_dim=N_EMBEDDING_DIM, latent_dim=N_LATENT_DIM, batch_dim=None, attention=ATTENTION, normalization=NORMALIZATION)

# COMPILE #####################################################################

MODEL.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=R_MAX),
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False, label_smoothing=0., axis=-1, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE, name='loss'),
    metrics=['accuracy'])

# TRAIN #######################################################################

tb_callback = tf.keras.callbacks.TensorBoard(log_dir=PATH_LOG)
cp_callback = tf.keras.callbacks.ModelCheckpoint(PATH_EXPORT, monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', save_freq='epoch')
lr_callback = tf.keras.callbacks.LearningRateScheduler(functools.partial(_mto.learning_rate_hokusai, lr_min=R_MIN, lr_max=R_MAX, lr_exp=R_EXP, rampup=N_EPOCHS_RAMPUP, sustain=N_EPOCHS_SUSTAIN), verbose=True)

if TRAINING:
    HISTORY = MODEL.fit(
        x=TRAIN['ar'].concatenate(TRAIN['en']).concatenate(TRAIN['es']).concatenate(TRAIN['de']).concatenate(TRAIN['hi']).concatenate(TRAIN['vi']).concatenate(TRAIN['zh']),
        batch_size=N_BATCH,
        epochs=N_EPOCHS,
        validation_split=None,
        validation_data=TEST['zh'], # full of glyphs
        validation_freq=list(range(1, N_EPOCHS + 1, N_EPOCHS // 8)),
        verbose=2,
        callbacks=[lr_callback, cp_callback, tb_callback])
