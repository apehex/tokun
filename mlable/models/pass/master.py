"""Tensorflow port of the tutorial by Andrej Karpathy: https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/"""

import datetime
import functools
import math
import os
import random

import tensorflow as tf

import mlable.inputs.vocabulary as _miv
import mlable.sampling as _ms

# META ########################################################################

N_VOCABULARY_DIM = 37
N_CONTEXT_DIM = 16 # necessary?
N_EMBEDDING_DIM = 128

N_BATCH = 128

N_PASSWORD = 256

# VOCABULARY ##################################################################

VOC_ALPHA_UPPER = ''.join(chr(__i) for __i in range(65, 91))    # A-Z
VOC_ALPHA_LOWER = VOC_ALPHA_UPPER.lower()                       # a-z
VOC_NUMBERS = '0123456789'                                      # 0-9
VOC_SYMBOLS = ''.join(chr(__i) for __i in range(33, 48))        # !"#$%&\'()*+,-./

def vocabulary(lower: bool=True, upper: bool=True, numbers: bool=True, symbols: bool=False) -> str:
    return sorted(set(lower * VOC_ALPHA_LOWER + upper * VOC_ALPHA_UPPER + numbers * VOC_NUMBERS + symbols * VOC_SYMBOLS))

VOCABULARY = vocabulary(1, 1, 1, 0)
N_VOCABULARY_DIM = len(VOCABULARY)

# MAPPINGS ####################################################################

MAPPINGS = _miv.mappings(vocabulary=VOCABULARY)

_stoi = MAPPINGS['encode']
_itos = MAPPINGS['decode']

# MODEL #######################################################################

def create_model(
    seed: int,
    n_context_dim: int=N_CONTEXT_DIM,
    n_vocabulary_dim: int=N_VOCABULARY_DIM,
    n_embedding_dim: int=N_EMBEDDING_DIM
) -> tf.keras.Model:
    __model = tf.keras.Sequential()
    # embedding
    __model.add(tf.keras.layers.Embedding(input_dim=n_vocabulary_dim, output_dim=n_embedding_dim, embeddings_initializer=tf.keras.initializers.glorot_uniform(seed=seed), name='embedding'))
    # head
    __model.add(tf.keras.layers.Dense(units=n_vocabulary_dim, activation='tanh', use_bias=False, kernel_initializer=tf.keras.initializers.glorot_uniform(seed=seed), name='head'))
    __model.add(tf.keras.layers.Softmax(axis=-1, name='softmax'))
    # build
    # __model(tf.keras.Input(shape=(n_context_dim,), batch_size=N_BATCH))
    # compile
    __model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_min),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False, label_smoothing=0., axis=-1, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE, name='loss'),
        metrics=['accuracy'])
    return __model

MODEL = create_model()

# INPUTS ######################################################################

# SAMPLE ######################################################################

def encode(input: str):
    return

def decode(output: list):
    return

def password(target: str, login: str, length: int, vocabulary: list, ):
    # login + target
    return ''

def sample(input: str, model: tf.keras.Model):
    return
