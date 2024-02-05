"""Tensorflow port of the tutorial by Andrej Karpathy: https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/"""

import argparse
import datetime
import functools
import itertools
import math
import os
import random

import tensorflow as tf

import mlable.inputs.vocabulary as _miv
import mlable.sampling as _ms

# DEFAULT INPUT VOCABULARY ####################################################

INPUT_VOCABULARY = ''.join(chr(__i) for __i in range(128)) # all ASCII characters

# DEFAULT OUTPUT VOCABULARY ###################################################

VOCABULARY_ALPHA_UPPER = ''.join(chr(__i) for __i in range(65, 91))                             # A-Z
VOCABULARY_ALPHA_LOWER = VOCABULARY_ALPHA_UPPER.lower()                                         # a-z
VOCABULARY_NUMBERS = '0123456789'                                                               # 0-9
VOCABULARY_SYMBOLS = ''.join(chr(__i) for __i in range(33, 48) if chr(__i) not in ["'", '"'])   # !#$%&\()*+,-./

OUTPUT_VOCABULARY = INPUT_VOCABULARY # placeholder

# DEFAULT META ################################################################

N_INPUT_DIM = len(INPUT_VOCABULARY) # all ASCII characters
N_OUTPUT_DIM = N_INPUT_DIM # placeholder, it depends on the user settings

N_CONTEXT_DIM = 8 # necessary?
N_EMBEDDING_DIM = 128

N_PASSWORD_DIM = 16
N_PASSWORD_NONCE = 1

# HYPER PARAMETERS ############################################################

def seed(key: str) -> int:
    __key = ''.join(__c for __c in key if ord(__c) < 128) # keep only ASCII characters
    return int(bytes(__key, 'utf-8').hex(), 16) % (2 ** 64) # dword

# VOCABULARY ##################################################################

def compose(lower: bool=True, upper: bool=True, numbers: bool=True, symbols: bool=False) -> str:
    return sorted(set(lower * VOCABULARY_ALPHA_LOWER + upper * VOCABULARY_ALPHA_UPPER + numbers * VOCABULARY_NUMBERS + symbols * VOCABULARY_SYMBOLS))

# MODEL #######################################################################

def create_model(
    seed: int,
    n_input_dim: int,
    n_output_dim: int,
    n_context_dim: int=N_CONTEXT_DIM,
    n_embedding_dim: int=N_EMBEDDING_DIM,
) -> tf.keras.Model:
    __model = tf.keras.Sequential()
    # embedding
    __model.add(tf.keras.layers.Embedding(input_dim=n_input_dim, output_dim=n_embedding_dim, embeddings_initializer=tf.keras.initializers.GlorotNormal(seed=seed), name='embedding'))
    # head
    __model.add(tf.keras.layers.Reshape(target_shape=(n_context_dim * n_embedding_dim,), input_shape=(n_context_dim, n_embedding_dim), name='reshape'))
    __model.add(tf.keras.layers.Dense(units=n_output_dim, activation='tanh', use_bias=False, kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seed), name='head'))
    __model.add(tf.keras.layers.Softmax(axis=-1, name='softmax'))
    # compile
    __model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False, label_smoothing=0., axis=-1, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE, name='loss'))
    return __model

# PREPROCESS ##################################################################

def remove_prefix(text: str) -> str:
    return

def remove_suffix(text: str) -> str:
    return

def preprocess(target: str, login: str, stoi: callable) -> list:
    __left = _miv.encode(text=target.lower(), stoi=stoi)
    __right = _miv.encode(text=login.lower(), stoi=stoi)
    return __left + __right

# ENTROPY #####################################################################

def accumulate(x: int, y: int, n: int) -> int:
    return (x + y) % n

def feed(source: list, nonce: int, dimension: int) -> iter:
    __func = lambda __x, __y: accumulate(x=__x, y=__y + nonce, n=dimension) # add entropy by accumulating the encodings
    return itertools.accumulate(iterable=itertools.cycle(source), func=__func) # infinite iterable

# INPUTS ######################################################################

def tensor(feed: 'Iterable[int]', length: int, context: int) -> tf.Tensor:
    __x = [[next(feed) for _ in range(context)] for _ in range(length)]
    return tf.constant(tf.convert_to_tensor(value=__x, dtype=tf.dtypes.int32))

# OUTPUTS #####################################################################

def password(model: tf.keras.Model, x: tf.Tensor, itos: callable) -> str:
    __y = tf.squeeze(model(x, training=False))
    __p = list(tf.argmax(__y, axis=-1).numpy())
    return _miv.decode(__p, itos=itos)

# PROCESS #####################################################################

def process(
    master_key: str,
    login_target: str,
    login_id: str,
    password_length: int,
    password_nonce: int,
    include_lower: bool,
    include_upper: bool,
    include_number: bool,
    include_symbol: bool,
    input_vocabulary: str=INPUT_VOCABULARY,
    model_context_dim: int=N_CONTEXT_DIM,
    model_embedding_dim: int=N_EMBEDDING_DIM
) -> str:
    # seed to generate the model weights randomly
    __seed = seed(key=master_key)
    # input vocabulary
    __input_mappings = _miv.mappings(vocabulary=input_vocabulary)
    __input_dim = len(input_vocabulary)
    # output vocabulary
    __output_vocabulary = compose(lower=include_lower, upper=include_upper, numbers=include_number, symbols=include_symbol)
    __output_mappings = _miv.mappings(vocabulary=__output_vocabulary)
    __output_dim = len(__output_vocabulary)
    # inputs
    __source = preprocess(target=login_target, login=login_id, stoi=__input_mappings['encode'])
    __feed = feed(source=__source, nonce=password_nonce, dimension=__input_dim)
    __x = tensor(feed=__feed, length=password_length, context=model_context_dim)
    # model
    __model = create_model(seed=__seed, n_input_dim=__input_dim, n_output_dim=__output_dim, n_context_dim=model_context_dim, n_embedding_dim=model_embedding_dim)
    # password
    __password = password(model=__model, x=__x, itos=__output_mappings['decode'])
    return __password

# CLI #########################################################################

# MAIN ########################################################################

process(master_key='yolomofo', login_target='huggingface.co', login_id='apehex', password_length=16, password_nonce=1, include_lower=True, include_upper=True, include_number=True, include_symbol=False, input_vocabulary=INPUT_VOCABULARY, model_context_dim=N_CONTEXT_DIM, model_embedding_dim=N_EMBEDDING_DIM)
