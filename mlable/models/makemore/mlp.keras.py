"""Tensorflow port of the tutorial by Andrej Karpathy: https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/"""

import datetime
import math
import os
import random

import tensorboard as tb
import tensorflow as tf

# META ########################################################################

N_ENCODING = 37
N_CONTEXT = 8
N_EMBEDDING = 64
N_HIDDEN = 512
N_SAMPLE = 32

N_EPOCHS = 16
N_BATCH = 1024

R_TRAINING = 0.001

VERSION = 'keras'

# N-GRAMS #####################################################################

def ngrams(word: str, length: int=N_CONTEXT):
    __context = length * '.'
    for __c in word + '.':
        yield __context
        __context = __context[1:] + __c

# ENCODING ####################################################################

def is_alpha(c: str):
    return ord(c.lower()) > 96 and ord(c.lower()) < 123

def is_num(c: str):
    return ord(c.lower()) > 47 and ord(c.lower()) < 58

def stoi(c: str) -> int:
    __i = 0
    if is_alpha(c):
        __i = ord(c.lower()) - 96
    if is_num(c):
        __i = 27 + ord(c.lower()) - 48
    return __i

def itos(i: int) -> str:
    __c = '.'
    if 0 < i and i < 27:
        __c = chr(i + 96)
    if 26 < i:
        __c = chr(i + 21)
    return __c

def encode(text: str) -> tf.Tensor:
    return [stoi(__c) for __c in text]

# DATASETS ####################################################################

def dataset(words: list, context: int=N_CONTEXT, depth: int=N_ENCODING) -> tuple:
    __x = [encode(__n) for __w in words for __n in ngrams(word=__w, length=context)]
    __y = [__i for __w in words for __i in encode(__w + '.')]
    return tf.convert_to_tensor(value=__x, dtype=tf.dtypes.int32), tf.one_hot(indices=__y, depth=depth, dtype=tf.dtypes.float32)

# MODEL #######################################################################

MODEL = tf.keras.Sequential()

# layers
MODEL.add(tf.keras.Input(shape=(N_CONTEXT,)))
MODEL.add(tf.keras.layers.Embedding(input_dim=N_ENCODING, output_dim=N_EMBEDDING, embeddings_initializer='he_normal', name='embedding'))
MODEL.add(tf.keras.layers.Reshape(target_shape=(N_CONTEXT * N_EMBEDDING,), name='reshape'))
MODEL.add(tf.keras.layers.Dense(units=N_HIDDEN, activation=None, use_bias=False, kernel_initializer='he_normal', name='hidden'))
MODEL.add(tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='normal', moving_mean_initializer='zeros', moving_variance_initializer='normal', name='normalization'))
MODEL.add(tf.keras.layers.Activation(activation='tanh', name='activation'))
MODEL.add(tf.keras.layers.Dense(units=N_ENCODING, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', name='head'))
MODEL.add(tf.keras.layers.Softmax(axis=-1, name='softmax'))

# compile
MODEL.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=R_TRAINING),
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False, label_smoothing=0., axis=-1, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE, name='loss'))

# SAMPLE ######################################################################

def tensor(ngram: list) -> tf.Tensor:
    return tf.convert_to_tensor(value=[ngram], dtype=tf.dtypes.int32)

def _next(model: tf.keras.Sequential, x: tf.Tensor, classes: int=N_ENCODING, highest: bool=False) -> int:
    __prob = model(x, training=False)[0]
    __unigrams = tf.cast(x=100. * __prob, dtype=tf.dtypes.int32).numpy().tolist()
    __highest = tf.argmax(__prob, axis=-1).numpy()
    __random, _, _ = tf.nn.fixed_unigram_candidate_sampler(
        true_classes = tf.convert_to_tensor([range(N_ENCODING)], dtype=tf.dtypes.int64),
        num_true = classes,
        num_sampled = 1,
        unique = False,
        range_max = classes,
        unigrams = __unigrams)
    return __highest if highest else __random.numpy()[0]

def sample(model: tf.keras.Sequential, context: int=N_CONTEXT, depth: int=N_ENCODING, max_length: int=N_SAMPLE) -> str:
    __i = 0
    __start = int(random.uniform(0, N_ENCODING))
    __result = itos(__start)
    __ngram = (context - 1) * [0,] + [__start]
    __x = tensor(ngram=__ngram)
    __n = _next(model=model, x=__x)
    while __n != 0 and __i < max_length:
        __ngram = __ngram[1:] + [__n]
        __x = tensor(ngram=__ngram)
        __n = _next(model=model, x=__x)
        __result += itos(__n)
        __i += 1
    return __result

# SAVE ########################################################################

# TEST ########################################################################

# DATA ########################################################################

USERNAMES = open('.data/usernames.txt', 'r').read().splitlines()

# filter non-ascii characters
USERNAMES = [__w for __w in USERNAMES if all([is_num(__c) or is_alpha(__c) for __c in __w])]

# randomize the order
random.shuffle(USERNAMES)

N1 = int(0.8 * len(USERNAMES))
N2 = int(0.9 * len(USERNAMES))

# SPLIT #######################################################################

X_TRAIN, Y_TRAIN = dataset(words=USERNAMES[:N1], context=N_CONTEXT)
X_DEV, Y_DEV = dataset(words=USERNAMES[N1:N2], context=N_CONTEXT)
X_TEST, Y_TEST = dataset(words=USERNAMES[N2:], context=N_CONTEXT)

# TRAIN #######################################################################

MODEL.fit(
    x=X_TRAIN,
    y=Y_TRAIN,
    batch_size=N_BATCH,
    epochs=N_EPOCHS,
    validation_split=None,
    validation_data=(X_DEV, Y_DEV),
    validation_freq=[1, N_EPOCHS],
    verbose=2)

# VIZ #########################################################################

# tf.argmax(X_TRAIN, axis=-1) # convert one_hot to indices and check the dataset
# tf.argmax(Y_TRAIN, axis=-1)

PATH = os.path.join('.logs/', VERSION, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
SUMMARY = tf.summary.create_file_writer(PATH)

# plot model stats
# save_model_histograms(model=MODEL, step=N_STEPS, summary=SUMMARY)

# plot loss
# save_loss_plot(data=L_TRAIN, name='train_loss', summary=SUMMARY, offset=0)
# save_loss_plot(data=L_TEST, name='test_loss', summary=SUMMARY, offset=0)

# plot log10(gradient / value)
# save_ratios_plot(data=G_RATIOS, model=MODEL, summary=SUMMARY, offset=0)
