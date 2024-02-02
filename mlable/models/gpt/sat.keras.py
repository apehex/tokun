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
import mlable.keras.models as _mkm
import mlable.tensorflow.summary as _mts

# META ########################################################################

N_VOCABULARY_DIM = 37
N_CONTEXT_DIM = 16
N_EMBEDDING_DIM = 64
N_ATTENTION_HEAD = 4
N_ATTENTION_DIM = N_EMBEDDING_DIM // N_ATTENTION_HEAD
N_HIDDEN_DIM = 4 * N_EMBEDDING_DIM # = 4 * N_ATTENTION_DIM * N_ATTENTION_HEAD

N_EPOCHS = 2
N_BATCH = 128

N_SAMPLE = 256

R_TRAINING = 0.0001

VERSION = 'sat-keras-125k'

# DATA ########################################################################

TEXT = open('.data/shakespeare/othello.md', 'r').read() # .splitlines()
TEXT += open('.data/shakespeare/hamlet.md', 'r').read() # .splitlines()

# VOCABULARY ##################################################################

VOCABULARY = _miv.capture(TEXT)
N_VOCABULARY_DIM = len(VOCABULARY)

# MAPPINGS ####################################################################

MAPPINGS = _miv.mappings(vocabulary=VOCABULARY, blank=_miv.BLANK)

_stoi = MAPPINGS['encode']
_itos = MAPPINGS['decode']

# MODEL #######################################################################

MODEL = tf.keras.Sequential()

# embedding
MODEL.add(tf.keras.layers.Embedding(input_dim=N_VOCABULARY_DIM, output_dim=N_EMBEDDING_DIM, embeddings_initializer='he_normal', name='embedding'))

# blocks
MODEL.add(_mkm.ResidualSelfAttentionDecoderBlock(hidden_dim=N_HIDDEN_DIM, attention_head_dim=N_ATTENTION_DIM, attention_head_count=N_ATTENTION_HEAD, normalization_epsilon=0.001, dropout=0.0, name='decoder-block-1'))

# head
MODEL.add(tf.keras.layers.Reshape(target_shape=(N_CONTEXT_DIM * N_EMBEDDING_DIM,), input_shape=(N_CONTEXT_DIM, N_EMBEDDING_DIM)))
MODEL.add(tf.keras.layers.Dense(units=N_VOCABULARY_DIM, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', name='head'))
MODEL.add(tf.keras.layers.Softmax(axis=-1, name='softmax'))

# build
# MODEL(tf.keras.Input(shape=(N_CONTEXT_DIM,), batch_size=N_BATCH))

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

X_TRAIN, Y_TRAIN = _min.dataset(text=TEXT[:N1], stoi=_stoi, context=N_CONTEXT_DIM, depth=N_VOCABULARY_DIM)
X_DEV, Y_DEV = _min.dataset(text=TEXT[N1:N2], stoi=_stoi, context=N_CONTEXT_DIM, depth=N_VOCABULARY_DIM)
X_TEST, Y_TEST = _min.dataset(text=TEXT[N2:], stoi=_stoi, context=N_CONTEXT_DIM, depth=N_VOCABULARY_DIM)

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

sample = functools.partial(_ms.sample, model=MODEL, context=N_CONTEXT_DIM, depth=N_VOCABULARY_DIM, length=N_SAMPLE, itos=_itos)

# VIZ #########################################################################

# tf.argmax(X_TRAIN, axis=-1) # convert one_hot to indices and check the dataset
# tf.argmax(Y_TRAIN, axis=-1)

# plot model stats
_mts.save_model_histograms(model=MODEL, epoch=N_EPOCHS, summary=SUMMARY)
