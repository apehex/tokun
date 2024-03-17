"""Tensorflow port of the tutorial by Andrej Karpathy: https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/"""

import datetime
import functools
import math
import os
import random

import tensorflow as tf

import mlable.tensorflow.data as _mtd
import mlable.tokens.ngrams as _mtn
import mlable.keras.models as _mkm
import mlable.tensorflow.sampling as _sam
import mlable.tensorflow.summary as _sum

# META ########################################################################

N_VOCABULARY_DIM = 37
N_CONTEXT_DIM = 16
N_EMBEDDING_DIM = 64
N_HIDDEN_DIM = 4 * N_EMBEDDING_DIM # = 4 * N_ATTENTION_DIM * N_ATTENTION_HEAD
N_ATTENTION_BLOCK = 1
N_ATTENTION_HEAD = 4
N_ATTENTION_DIM = N_EMBEDDING_DIM // N_ATTENTION_HEAD

N_EPOCHS = 8
N_EPOCHS_RAMPUP = 4
N_EPOCHS_SUSTAIN = 0

N_BATCH = 128

N_SAMPLE = 256

R_MIN = 0.00001
R_MAX = 0.0001
R_EXP = .8

VERSION = 'sat-keras-125k'

# DATA ########################################################################

TEXT = open('.data/shakespeare/othello.md', 'r').read() # .splitlines()
TEXT += open('.data/shakespeare/hamlet.md', 'r').read() # .splitlines()

# VOCABULARY ##################################################################

VOCABULARY = _mtn.vocabulary(TEXT)
N_VOCABULARY_DIM = len(VOCABULARY)

# MAPPINGS ####################################################################

MAPPINGS = _mtn.mappings(voc=VOCABULARY)

_stoi = MAPPINGS['encode']
_itos = MAPPINGS['decode']

# MODEL #######################################################################

def create_model(
    n_context_dim: int=N_CONTEXT_DIM,
    n_vocabulary_dim: int=N_VOCABULARY_DIM,
    n_embedding_dim: int=N_EMBEDDING_DIM,
    n_hidden_dim: int=N_HIDDEN_DIM,
    n_attention_block: int=N_ATTENTION_BLOCK,
    n_attention_head: int=N_ATTENTION_HEAD,
    n_attention_dim: int=N_ATTENTION_DIM,
    lr_mtn: float=R_MIN
) -> tf.keras.Model:
    __model = tf.keras.Sequential()
    # embedding
    __model.add(tf.keras.layers.Embedding(input_dim=n_vocabulary_dim, output_dim=n_embedding_dim, embeddings_initializer='he_normal', name='embedding'))
    # blocks
    for __i in range(n_attention_block):
        __model.add(_mkm.ResidualSelfAttentionDecoderBlock(hidden_dim=n_hidden_dim, attention_head_dim=n_attention_dim, attention_head_count=n_attention_head, normalization_epsilon=0.001, dropout=0.0, name='decoder-block-' + str(__i)))
    # head
    __model.add(tf.keras.layers.Reshape(target_shape=(n_context_dim * n_embedding_dim,), input_shape=(n_context_dim, n_embedding_dim), name='reshape'))
    __model.add(tf.keras.layers.Dense(units=n_vocabulary_dim, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', name='head'))
    __model.add(tf.keras.layers.Softmax(axis=-1, name='softmax'))
    # build
    # __model(tf.keras.Input(shape=(n_context_dim,), batch_size=N_BATCH))
    # compile
    __model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_mtn),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False, label_smoothing=0., axis=-1, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE, name='loss'),
        metrics=['accuracy'])
    return __model

MODEL = create_model()

# SAVE ########################################################################

# log path
LOGPATH = os.path.join('.logs/', VERSION, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
SUMMARY = tf.summary.create_file_writer(LOGPATH)

# called during training
tb_callback = tf.keras.callbacks.TensorBoard(log_dir=LOGPATH)

# SPLIT #######################################################################

N1 = int(0.8 * len(TEXT))
N2 = int(0.9 * len(TEXT))

__x, __y = _mtn.tokenize(text=TEXT, stoi=_stoi, context_length=N_CONTEXT_DIM)
__X, __Y = _mtd.dataset(x=__x, y=__y, depth=N_VOCABULARY_DIM)

X_TRAIN, Y_TRAIN = __X[:N1], __Y[:N1]
X_DEV, Y_DEV = __X[N1:N2], __Y[N1:N2]
X_TEST, Y_TEST = __X[N2:], __Y[N2:]

# LEARNING RATE ###############################################################

def lrfn(epoch: int, lr_mtn: float, lr_max: float, lr_exp: float, rampup: int, sustain: int) -> float:
    __lr = lr_mtn
    if epoch < rampup:
        __lr = lr_mtn + (epoch * (lr_max - lr_mtn) / rampup)
    elif epoch < rampup + sustain:
        __lr = lr_max
    else:
        __lr = lr_mtn + (lr_max - lr_mtn) * lr_exp ** (epoch - rampup - sustain)
    return __lr

lr_callback = tf.keras.callbacks.LearningRateScheduler(lambda epoch: lrfn(epoch, lr_mtn=R_MIN, lr_max=R_MAX, lr_exp=R_EXP, rampup=N_EPOCHS_RAMPUP, sustain=N_EPOCHS_SUSTAIN), verbose=True)

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
    callbacks=[lr_callback, tb_callback])

# SAMPLE ######################################################################

sample = functools.partial(_sam.sample, model=MODEL, context=N_CONTEXT_DIM, depth=N_VOCABULARY_DIM, length=N_SAMPLE, itos=_itos)

# VIZ #########################################################################

# tf.argmax(X_TRAIN, axis=-1) # convert one_hot to indices and check the dataset
# tf.argmax(Y_TRAIN, axis=-1)

# plot model stats
_sum.save_model_histograms(model=MODEL, epoch=N_EPOCHS, summary=SUMMARY)
