"""Tensorflow port of the tutorial by Andrej Karpathy: https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/"""

import datetime
import math
import os
import random

import tensorboard as tb
import tensorflow as tf

# META ########################################################################

N_ENCODING = 37
N_CONTEXT = 4
N_EMBEDDING = 64
N_HIDDEN = 256
N_SAMPLE = 32

N_STEPS = 1024
N_BATCH = 4096

G_REGULARIZATION = 1.

R_TRAINING = 1.

VERSION = 'batch-norm'

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
    return tf.one_hot(indices=__x, depth=depth, dtype=tf.dtypes.float32), tf.one_hot(indices=__y, depth=depth, dtype=tf.dtypes.float32)

# MODEL #######################################################################

class Model(tf.Module):
    def __init__(self, n_context: int=N_CONTEXT, n_encoding: int=N_ENCODING, n_embedding: int=N_EMBEDDING, n_hidden: int=N_HIDDEN, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # save the parameters
        self._N_CONTEXT = n_context
        self._N_ENCODING = n_encoding
        self._N_EMBEDDING = n_embedding
        self._N_HIDDEN = n_hidden
        # embedding
        self._C = tf.Variable(initial_value=tf.random.normal(shape=(self._N_ENCODING, self._N_EMBEDDING), mean=0., stddev=1., dtype=tf.dtypes.float32), name='C')
        # hidden layer
        self._W1 = tf.Variable(initial_value=0.1 * tf.random.normal(shape=(self._N_EMBEDDING * self._N_CONTEXT, self._N_HIDDEN), mean=0., stddev=1., dtype=tf.dtypes.float32), name='W1')
        # normalization layer
        self._M1 = tf.Variable(initial_value=tf.zeros(shape=(1, self._N_HIDDEN), dtype=tf.dtypes.float32), name='M1')
        self._S1 = tf.Variable(initial_value=tf.ones(shape=(1, self._N_HIDDEN), dtype=tf.dtypes.float32), name='S1')
        self._G1 = tf.Variable(initial_value=tf.ones(shape=(1, self._N_HIDDEN), dtype=tf.dtypes.float32), name='G1')
        self._B1 = tf.Variable(initial_value=0.1 * tf.random.normal(shape=(1, self._N_HIDDEN), mean=0., stddev=1., dtype=tf.dtypes.float32), name='B1')
        # head layer
        self._W2 = tf.Variable(initial_value=0.1 * tf.random.normal(shape=(self._N_HIDDEN, self._N_ENCODING), mean=0., stddev=1., dtype=tf.dtypes.float32), name='W2')
        self._B2 = tf.Variable(initial_value=0.1 * tf.random.normal(shape=(1, self._N_ENCODING), mean=0., stddev=1., dtype=tf.dtypes.float32), name='B2')
        # parameters
        self.parameters = [self._C, self._W1, self._W2, self._B1, self._B2, self._G1]

    def __call__(self, x):
        # embed the input vector / matrix
        __e = tf.reshape(x @ self._C, (x.shape[0], self._N_CONTEXT * self._N_EMBEDDING))
        # hidden layer
        __i = __e @ self._W1 # no need for a bias
        # normalize
        __i_mean = tf.math.reduce_mean(__i, axis=0, keepdims=True)
        __i_std = tf.math.reduce_std(__i, axis=0, keepdims=True)
        self._M1 = tf.stop_gradient(0.999 * self._M1 + 0.001 * __i_mean)
        self._S1 = tf.stop_gradient(0.999 * self._S1 + 0.001 * __i_std)
        __n = tf.math.divide(__i - self._M1, self._S1)
        # activation
        __h = tf.math.tanh(tf.math.multiply(self._G1, __n) + self._B1)
        # head layer
        return tf.nn.softmax(__h @ self._W2 + self._B2)

# LOSS ########################################################################

def loss(target_y: tf.Tensor, predicted_y: tf.Tensor):
    __l = tf.keras.losses.CategoricalCrossentropy(from_logits=False, label_smoothing=0., axis=-1, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE, name='L')
    # __p = tf.math.multiply(x=target_y, y=1. - predicted_y) # low probability = high loss
    return __l(target_y, predicted_y)

# SAMPLE ######################################################################

def tensor(ngram: list, depth: int=N_ENCODING) -> tf.Tensor:
    return tf.one_hot(indices=[ngram], depth=depth, dtype=tf.dtypes.float32)

def _next(model: Model, x: tf.Tensor, classes: int=N_ENCODING, rand: bool=True) -> int:
    __prob = model(x)[0]
    __unigrams = tf.cast(x=100. * __prob, dtype=tf.dtypes.int32).numpy().tolist()
    __highest = tf.argmax(__prob, axis=-1).numpy()
    __random, _, _ = tf.nn.fixed_unigram_candidate_sampler(
        true_classes = tf.convert_to_tensor([range(N_ENCODING)], dtype=tf.dtypes.int64),
        num_true = classes,
        num_sampled = 1,
        unique = False,
        range_max = classes,
        unigrams = __unigrams)
    return __random.numpy()[0] if rand else __highest

def sample(model: Model, context: int=N_CONTEXT, depth: int=N_ENCODING, max_length: int=N_SAMPLE) -> str:
    __i = 0
    __start = int(random.uniform(0, N_ENCODING))
    __result = itos(__start)
    __ngram = (context - 1) * [0,] + [__start]
    __x = tensor(ngram=__ngram, depth=depth)
    __n = _next(model=model, x=__x)
    while __n != 0 and __i < max_length:
        __ngram = __ngram[1:] + [__n]
        __x = tensor(ngram=__ngram, depth=depth)
        __n = _next(model=model, x=__x)
        __result += itos(__n)
        __i += 1
    return __result

# TRAIN #######################################################################

def step(model: Model, x: tf.Tensor, y: tf.Tensor, rate: float=R_TRAINING) -> tuple:
    with tf.GradientTape() as __tape:
        # grad / data
        __ratios = []
        # compute gradient on these
        __tape.watch(model.parameters)
        # forward
        __loss = loss(y, model(x))
        # backward
        __grad = __tape.gradient(__loss, model.parameters)
        # update the model
        for __i in range(len(model.parameters)):
            model.parameters[__i].assign_sub(rate * __grad[__i])
            __ratios.append(math.log10(tf.math.reduce_std(rate * __grad[__i]) / tf.math.reduce_std(model.parameters[__i])))
    return __loss, __ratios

def train(model: Model, x_train: tf.Tensor, y_train: tf.Tensor, x_test: tf.Tensor, y_test:tf.Tensor, steps: int=N_STEPS, batch: int=N_BATCH, rate: float=R_TRAINING) -> tuple:
    __n_variables = len(model.parameters)
    __train_loss = []
    __test_loss = []
    __ratios = []
    for __i in range(steps + 1):
        # random batch
        __indices = random.sample(population=range(x_train.shape[0]), k=batch)
        __x = tf.gather(params=x_train, indices=__indices, axis=0)
        __y = tf.gather(params=y_train, indices=__indices, axis=0)
        # update the model
        __loss, __r = step(model=model, x=__x, y=__y, rate=rate)
        # save loss
        __train_loss.append((__i, __loss))
        # save ratios grad / data
        __ratios.append(__r)
        # log the progress
        if __i % int(0.1 * steps) == 0:
            __test_loss.append((__i, loss(target_y=y_test, predicted_y=model(x_test))))
            print('[epoch {epoch}] train loss: {train} test loss: {test}'.format(epoch=__i, train=__train_loss[-1][1], test=__test_loss[-1][1]))
    return __train_loss, __test_loss, __ratios

# SAVE ########################################################################

def save_model_histograms(model: Model, step: int, summary: 'ResourceSummaryWriter') -> None:
    with summary.as_default():
        for __p in model.parameters:
            tf.summary.histogram(__p.name, __p, step=step)

def save_loss_plot(data: list, name: str, summary: 'ResourceSummaryWriter') -> None:
    with summary.as_default():
        for __i, __l in data:
            tf.summary.scalar(name, __l, step=__i)

def save_ratios_plot(data: list, model: Model, summary: 'ResourceSummaryWriter') -> None:
    with summary.as_default():
        for __i, __ratios in enumerate(data):
            for __j, __r in enumerate(__ratios):
                tf.summary.scalar(model.parameters[__j].name + '_log10(gradient/value)', __r, step=__i)

# TEST ########################################################################

# DATA ########################################################################

USERNAMES = open('.data/usernames.txt', 'r').read().splitlines()

# filter non-ascii characters
USERNAMES = [__w for __w in USERNAMES if all([is_num(__c) or is_alpha(__c) for __c in __w])]

# randomize the order
random.shuffle(USERNAMES)

# SPLIT #######################################################################

N1 = int(0.8 * len(USERNAMES))
N2 = int(0.9 * len(USERNAMES))

X_TRAIN, Y_TRAIN = dataset(words=USERNAMES[:N1], context=N_CONTEXT)
X_DEV, Y_DEV = dataset(words=USERNAMES[N1:N2], context=N_CONTEXT)
X_TEST, Y_TEST = dataset(words=USERNAMES[N2:], context=N_CONTEXT)

# MAIN ########################################################################

MODEL = Model()
L_TRAIN, L_TEST, G_RATIOS = train(model=MODEL, x_train=X_TRAIN, y_train=Y_TRAIN, x_test=X_TEST, y_test=Y_TEST, steps=N_STEPS, batch=N_BATCH, rate=R_TRAINING)

# VIZ #########################################################################

# tf.argmax(X_TRAIN, axis=-1) # convert one_hot to indices and check the dataset
# tf.argmax(Y_TRAIN, axis=-1)

PATH = os.path.join('.logs/', VERSION, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
SUMMARY = tf.summary.create_file_writer(PATH)

# plot model stats
save_model_histograms(model=MODEL, step=N_STEPS, summary=SUMMARY)

# plot loss
save_loss_plot(data=L_TRAIN, name='train_loss', summary=SUMMARY)
save_loss_plot(data=L_TEST, name='test_loss', summary=SUMMARY)

# plot log10(gradient / value)
save_ratios_plot(data=G_RATIOS, model=MODEL, summary=SUMMARY)
