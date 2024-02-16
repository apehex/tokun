"""Tensorflow port of the tutorial by Andrej Karpathy: https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/"""

import datetime
import math
import os
import random

import tensorflow as tf

# META ########################################################################

N_ENCODING = 37
N_CONTEXT = 8
N_EMBEDDING = 32
N_HIDDEN = 256
N_SAMPLE = 32

N_STEPS = 1024
N_BATCH = 4096

R_TRAINING = 0.2

VERSION = 'mlp-tf-80k'

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

# INITIALIZER #################################################################

class SmallNormal(tf.keras.initializers.Initializer):
    def __init__(self, mean: float=0., stddev: float=0.1):
        self._mean = 0.
        self._stddev = stddev

    def __call__(self, shape, dtype=None, **kwargs):
        return tf.random.normal(shape, mean=self._mean, stddev=self._stddev, dtype=dtype)

    def get_config(self):  # To support serialization
        return {"mean": self._mean, "stddev": self._stddev}

# LAYERS ######################################################################

class Activation(tf.keras.layers.Layer):
    def __init__(
        self,
        function: callable,
        **kwargs
    ):
        super(Activation, self).__init__(**kwargs)
        self._function = function

    def call(self, inputs: tf.Tensor, **kwargs):
        return self._function(inputs)

class BatchNormalization(tf.keras.layers.Layer):
    def __init__(
        self,
        axis=-1,
        momentum=0.99,
        epsilon=0.001,
        **kwargs
    ):
        super(BatchNormalization, self).__init__(**kwargs)
        self._axis = axis
        self._momentum = momentum
        self._epsilon = epsilon
        self._mean = None
        self._stddev = None
        self._gain = None
        self._bias = None

    def build(self, shape: tuple):
        # shape
        __axis = self._axis % len(shape) # positive index even when the axis is specified negatively, like -2
        __shape = [__d for __i, __d in enumerate(shape) if __i != __axis]
        # values
        __mean_init = SmallNormal()
        __stddev_init = SmallNormal()
        __gain_init = SmallNormal()
        __bias_init = SmallNormal()
        # tensors
        self._mean = self.add_weight("mean", shape=__shape, initializer=__mean_init)
        self._stddev = self.add_weight("stddev", shape=__shape, initializer=__stddev_init)
        self._gain = self.add_weight("gain", shape=__shape, initializer=__gain_init)
        self._bias = self.add_weight("bias", shape=__shape, initializer=__bias_init)

    def call(self, inputs: tf.Tensor, training: bool=True, **kwargs):
        if training:
            # current values
            __batch_mean = tf.math.reduce_mean(inputs, axis=self._axis, keepdims=True)
            __batch_stddev = tf.math.reduce_std(inputs, axis=self._axis, keepdims=True)
            # update parameters
            self._mean = tf.stop_gradient(self._momentum * self._mean + (1. - self._momentum) * __batch_mean)
            self._stddev = tf.stop_gradient(self._momentum * self._stddev + (1. - self._momentum) * __batch_stddev)
        # normalize
        __normalized = tf.math.divide(inputs - self._mean, self._stddev + self._epsilon)
        # scale
        return tf.math.multiply(self._gain, __normalized) + self._bias

class Dense(tf.keras.layers.Layer):
    def __init__(
        self,
        units: int,
        use_bias: bool=True,
        **kwargs
    ):
        super(Dense, self).__init__(**kwargs)
        self._units = units
        self._biased = use_bias
        self._kernel = None
        self._bias = None

    def build(self, shape: tuple):
        # kernel
        __kernel_init = SmallNormal()
        self._kernel = self.add_weight("kernel", shape=[int(shape[-1]), self._units], initializer=__kernel_init)
        # bias
        if self._biased:
            __bias_init = SmallNormal()
            self._bias = self.add_weight("bias", shape=[self._units], initializer=__bias_init)

    def call(self, inputs: tf.Tensor, **kwargs):
        return tf.matmul(inputs, self._kernel) + self._bias if (self._biased and self._bias is not None) else tf.matmul(inputs, self._kernel)

class Embedding(Dense):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        **kwargs
    ):
        super(Embedding, self).__init__(units=output_dim, use_bias=False, **kwargs)
        self._input_dim = input_dim
        self._output_dim = output_dim

    def build(self, shape: tuple):
        __shape = list(shape)
        # add a direction for the one-hot encoding
        __shape = __shape + [self._input_dim]
        # init
        super(Embedding, self).build(shape=__shape)

    def call(self, inputs: tf.Tensor, **kwargs):
        __x = tf.one_hot(indices=inputs, depth=self._input_dim, dtype=tf.dtypes.float32)
        return super(Embedding, self).call(inputs=__x, **kwargs)

class Reshape(tf.keras.layers.Layer):
    def __init__(
        self,
        target_shape: tuple,
        **kwargs
    ):
        super(Reshape, self).__init__(**kwargs)
        self._shape = target_shape

    def call(self, inputs: tf.Tensor, **kwargs):
        return tf.reshape(inputs, self._shape)

class Softmax(tf.keras.layers.Layer):
    def __init__(
        self,
        axis: int=-1,
        **kwargs
    ):
        super(Softmax, self).__init__(**kwargs)
        self._axis = axis

    def call(self, inputs: tf.Tensor, **kwargs):
        return tf.nn.softmax(inputs, axis=self._axis)

# MODEL #######################################################################

class Model(tf.Module):
    def __init__(self, n_context: int=N_CONTEXT, n_encoding: int=N_ENCODING, n_embedding: int=N_EMBEDDING, n_hidden: int=N_HIDDEN, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # layers
        self._layers = [
            Embedding(input_dim=n_encoding, output_dim=n_embedding, name='embedding'),
            Reshape(target_shape=(-1, n_context * n_embedding), name='reshape'),
            Dense(units=n_hidden, use_bias=False, name='hidden'),
            BatchNormalization(axis=0, name='normalization'),
            Activation(function=tf.math.tanh, name='activation'),
            Dense(units=n_encoding, use_bias=True, name='head'),
            Softmax(axis=-1, name='softmax')]

    def __call__(self, x, training: bool=True):
        __x = x
        # propagate x
        for __l in self._layers:
            __x = __l(__x, training=training)
        # return the output of the latest layer
        return __x

    def n_trainable_elements(self):
        return sum([tf.size(__v).numpy() for __v in MODEL.trainable_variables])

# LOSS ########################################################################

def loss(target_y: tf.Tensor, predicted_y: tf.Tensor):
    __l = tf.keras.losses.CategoricalCrossentropy(from_logits=False, label_smoothing=0., axis=-1, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE, name='loss')
    return __l(target_y, predicted_y)

# SAMPLE ######################################################################

def tensor(ngram: list) -> tf.Tensor:
    return tf.convert_to_tensor(value=[ngram], dtype=tf.dtypes.int32)

def _next(model: Model, x: tf.Tensor, classes: int=N_ENCODING, highest: bool=False) -> int:
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

def sample(model: Model, context: int=N_CONTEXT, depth: int=N_ENCODING, max_length: int=N_SAMPLE) -> str:
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

# TRAIN #######################################################################

def step(model: Model, x: tf.Tensor, y: tf.Tensor, rate: float=R_TRAINING) -> tuple:
    with tf.GradientTape() as __tape:
        # grad / data
        __ratios = []
        # compute gradient on these
        __tape.watch(model.trainable_variables)
        # loss
        __loss = loss(target_y=y, predicted_y=model(x, training=True))
        # backward
        __grad = __tape.gradient(__loss, model.trainable_variables)
        # update the model
        for __i in range(len(model.trainable_variables)):
            model.trainable_variables[__i].assign_sub(rate * __grad[__i])
            __ratios.append(math.log10(tf.math.reduce_std(rate * __grad[__i]) / tf.math.reduce_std(model.trainable_variables[__i])))
    return __loss, __ratios

def train(model: Model, x_train: tf.Tensor, y_train: tf.Tensor, x_test: tf.Tensor, y_test:tf.Tensor, steps: int=N_STEPS, batch: int=N_BATCH, rate: float=R_TRAINING) -> tuple:
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
            __test_loss.append((__i, loss(target_y=y_test, predicted_y=model(x_test, training=False))))
            print('[epoch {epoch}] train loss: {train} test loss: {test}'.format(epoch=__i, train=__train_loss[-1][1], test=__test_loss[-1][1]))
    return __train_loss, __test_loss, __ratios

# SAVE ########################################################################

def save_model_histograms(model: Model, step: int, summary: 'ResourceSummaryWriter') -> None:
    with summary.as_default():
        for __p in model.variables:
            tf.summary.histogram(__p.name, __p, step=step)

def save_loss_plot(data: list, name: str, summary: 'ResourceSummaryWriter', offset: int=0) -> None:
    with summary.as_default():
        for __i, __l in data:
            tf.summary.scalar(name, __l, step=__i + offset)

def save_ratios_plot(data: list, model: Model, summary: 'ResourceSummaryWriter', offset: int=0) -> None:
    with summary.as_default():
        for __i, __ratios in enumerate(data):
            for __j, __r in enumerate(__ratios):
                tf.summary.scalar(model.trainable_variables[__j].name + '_log10(gradient/value)', __r, step=__i + offset)

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
save_loss_plot(data=L_TRAIN, name='train_loss', summary=SUMMARY, offset=0)
save_loss_plot(data=L_TEST, name='test_loss', summary=SUMMARY, offset=0)

# plot log10(gradient / value)
save_ratios_plot(data=G_RATIOS, model=MODEL, summary=SUMMARY, offset=0)
