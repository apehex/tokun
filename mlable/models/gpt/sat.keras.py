"""Tensorflow port of the tutorial by Andrej Karpathy: https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/"""

import datetime
import functools
import math
import os
import random

import tensorflow as tf

import mlable.sampling as _ms
import mlable.inputs.ngrams as _min
import mlable.inputs.vocabulary as _miv
import mlable.tensorflow.summary as _mts

# META ########################################################################

N_VOCABULARY = 37
N_CONTEXT = 32
N_EMBEDDING = 32
N_HIDDEN = 128
N_SAMPLE = 256

N_EPOCHS = 16
N_BATCH = 128

R_TRAINING = 0.0001

VERSION = 'sat-keras-80k'

# DATA ########################################################################

TEXT = open('.data/shakespeare/othello.md', 'r').read() # .splitlines()
TEXT += open('.data/shakespeare/hamlet.md', 'r').read() # .splitlines()

# VOCABULARY ##################################################################

VOCABULARY = _miv.capture(TEXT)
N_VOCABULARY = len(VOCABULARY)

# MAPPINGS ####################################################################

MAPPINGS = _miv.mappings(vocabulary=VOCABULARY, blank='$')

_stoi = MAPPINGS['encode']
_itos = MAPPINGS['decode']

# MODEL #######################################################################

MODEL = tf.keras.Sequential()

# layers
MODEL.add(tf.keras.Input(shape=(N_CONTEXT,)))

# embedding
MODEL.add(tf.keras.layers.Embedding(input_dim=N_VOCABULARY, output_dim=N_EMBEDDING, embeddings_initializer='he_normal', name='embedding'))

# block 1
MODEL.add(tf.keras.layers.Reshape(target_shape=(N_CONTEXT // 2, 2 * N_EMBEDDING,), name='merge-2'))
MODEL.add(tf.keras.layers.Dense(units=N_HIDDEN, activation=None, use_bias=False, kernel_initializer='he_normal', name='compress-2'))
MODEL.add(tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='normal', moving_mean_initializer='zeros', moving_variance_initializer='normal', name='normalization-2'))
MODEL.add(tf.keras.layers.Activation(activation='tanh', name='activation-2'))

# block 2
MODEL.add(tf.keras.layers.Reshape(target_shape=(N_CONTEXT // 4, 2 * N_HIDDEN,), name='merge-4'))
MODEL.add(tf.keras.layers.Dense(units=N_HIDDEN, activation=None, use_bias=False, kernel_initializer='he_normal', name='compress-4'))
MODEL.add(tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='normal', moving_mean_initializer='zeros', moving_variance_initializer='normal', name='normalization-4'))
MODEL.add(tf.keras.layers.Activation(activation='tanh', name='activation-4'))

# block 3
MODEL.add(tf.keras.layers.Reshape(target_shape=(N_CONTEXT // 8, 2 * N_HIDDEN,), name='merge-8'))
MODEL.add(tf.keras.layers.Dense(units=N_HIDDEN, activation=None, use_bias=False, kernel_initializer='he_normal', name='compress-8'))
MODEL.add(tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='normal', moving_mean_initializer='zeros', moving_variance_initializer='normal', name='normalization-8'))
MODEL.add(tf.keras.layers.Activation(activation='tanh', name='activation-8'))

# block 4
MODEL.add(tf.keras.layers.Reshape(target_shape=(N_CONTEXT // 16, 2 * N_HIDDEN,), name='merge-16'))
MODEL.add(tf.keras.layers.Dense(units=N_HIDDEN, activation=None, use_bias=False, kernel_initializer='he_normal', name='compress-16'))
MODEL.add(tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='normal', moving_mean_initializer='zeros', moving_variance_initializer='normal', name='normalization-16'))
MODEL.add(tf.keras.layers.Activation(activation='tanh', name='activation-16'))

# block 4
MODEL.add(tf.keras.layers.Reshape(target_shape=(2 * N_HIDDEN,), name='merge-32'))
MODEL.add(tf.keras.layers.Dense(units=N_HIDDEN, activation=None, use_bias=False, kernel_initializer='he_normal', name='compress-32'))
MODEL.add(tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='normal', moving_mean_initializer='zeros', moving_variance_initializer='normal', name='normalization-32'))
MODEL.add(tf.keras.layers.Activation(activation='tanh', name='activation-32'))

# head
MODEL.add(tf.keras.layers.Dense(units=N_VOCABULARY, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', name='head'))
MODEL.add(tf.keras.layers.Softmax(axis=-1, name='softmax'))

# compile
MODEL.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=R_TRAINING),
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False, label_smoothing=0., axis=-1, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE, name='loss'))

# SAVE ########################################################################

# log path
LOGPATH = os.path.join('.logs/', VERSION, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
SUMMARY = tf.summary.create_file_writer(LOGPATH)

# called during training
CALLBACK = tf.keras.callbacks.TensorBoard(log_dir=LOGPATH)

# SPLIT #######################################################################

N1 = int(0.8 * len(TEXT))
N2 = int(0.9 * len(TEXT))

X_TRAIN, Y_TRAIN = _min.dataset(text=TEXT[:N1], stoi=_stoi, context=N_CONTEXT, depth=N_VOCABULARY)
X_DEV, Y_DEV = _min.dataset(text=TEXT[N1:N2], stoi=_stoi, context=N_CONTEXT, depth=N_VOCABULARY)
X_TEST, Y_TEST = _min.dataset(text=TEXT[N2:], stoi=_stoi, context=N_CONTEXT, depth=N_VOCABULARY)

# TRAIN #######################################################################

TRAINING_HISTORY = MODEL.fit(
    x=X_TRAIN,
    y=Y_TRAIN,
    batch_size=N_BATCH,
    epochs=N_EPOCHS,
    validation_split=None,
    validation_data=(X_DEV, Y_DEV),
    validation_freq=[1, N_EPOCHS],
    verbose=2,
    callbacks=[CALLBACK])

# SAMPLE ######################################################################

sample = functools.partial(_ms.sample, model=MODEL, context=N_CONTEXT, depth=N_VOCABULARY, length=N_SAMPLE, itos=_itos)

# VIZ #########################################################################

# tf.argmax(X_TRAIN, axis=-1) # convert one_hot to indices and check the dataset
# tf.argmax(Y_TRAIN, axis=-1)

# plot model stats
_mts.save_model_histograms(model=MODEL, epoch=N_EPOCHS, summary=SUMMARY)
