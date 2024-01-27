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
import mlable.tensorflow.layers as _mtl
import mlable.tensorflow.optimizers as _mto
import mlable.tensorflow.summary as _mts

# META ########################################################################

N_VOCABULARY = 37
N_CONTEXT = 8
N_EMBEDDING = 32
N_HIDDEN = 128
N_SAMPLE = 256

N_STEPS = 1024
N_BATCH = 128

R_TRAINING = 0.1

VERSION = 'sat-tf-80k'

# DATA ########################################################################

TEXT = open('.data/shakespeare/othello.md', 'r').read() # .splitlines()
TEXT += open('.data/shakespeare/hamlet.md', 'r').read() # .splitlines()

# randomize the order
# random.shuffle(TEXT)

# VOCABULARY ##################################################################

VOCABULARY = _miv.capture(TEXT)
N_VOCABULARY = len(VOCABULARY)

# MAPPINGS ####################################################################

MAPPINGS = _miv.mappings(vocabulary=VOCABULARY, blank='$')

_stoi = MAPPINGS['encode']
_itos = MAPPINGS['decode']

# DATASETS ####################################################################

def dataset(text: list, stoi: callable=_stoi, context: int=N_CONTEXT, depth: int=N_VOCABULARY) -> tuple:
    __x = [_miv.encode(text=__n, stoi=stoi) for __n in _min.tokenize(text=text, length=context)]
    __y = _miv.encode(text=text, stoi=stoi)
    return tf.convert_to_tensor(value=__x, dtype=tf.dtypes.int32), tf.one_hot(indices=__y, depth=depth, dtype=tf.dtypes.float32)

# MODEL #######################################################################

class Model(tf.Module):
    def __init__(self, n_context: int=N_CONTEXT, N_VOCABULARY: int=N_VOCABULARY, n_embedding: int=N_EMBEDDING, n_hidden: int=N_HIDDEN, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # layers
        self._layers = [
            # embedding
            _mtl.Embedding(input_dim=N_VOCABULARY, output_dim=n_embedding, name='embedding'),
            # block 1
            _mtl.Merge(axis=1, n=2, name='merge-2'),
            _mtl.Dense(units=n_hidden, use_bias=False, name='hidden-2'),
            _mtl.BatchNormalization(axis=0, name='normalization-2'),
            _mtl.Activation(function=tf.math.tanh, name='activation-2'),
            # block 2
            _mtl.Merge(axis=1, n=2, name='merge-4'),
            _mtl.Dense(units=n_hidden, use_bias=False, name='hidden-4'),
            _mtl.BatchNormalization(axis=0, name='normalization-4'),
            _mtl.Activation(function=tf.math.tanh, name='activation-4'),
            # block 3
            _mtl.Merge(axis=1, n=2, name='merge-8'),
            _mtl.Dense(units=n_hidden, use_bias=False, name='hidden-8'),
            _mtl.BatchNormalization(axis=0, name='normalization-8'),
            _mtl.Activation(function=tf.math.tanh, name='activation-8'),
            # head
            _mtl.Dense(units=N_VOCABULARY, use_bias=True, name='head'),
            _mtl.Softmax(axis=-1, name='softmax')]

    def __call__(self, x, training: bool=True):
        __x = x
        # propagate x
        for __l in self._layers:
            __x = __l(__x, training=training)
        # return the output of the latest layer
        return __x

    def n_trainable_elements(self):
        return tf.reduce_sum([tf.size(__v) for __v in MODEL.trainable_variables]).numpy()

# LOSS ########################################################################

def loss(target_y: tf.Tensor, predicted_y: tf.Tensor):
    __l = tf.keras.losses.CategoricalCrossentropy(from_logits=False, label_smoothing=0., axis=-1, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE, name='loss')
    return __l(target_y, predicted_y)

# TEST ########################################################################

# DATA ########################################################################

N1 = int(0.8 * len(TEXT))
N2 = int(0.9 * len(TEXT))

X_TRAIN, Y_TRAIN = dataset(text=TEXT[:N1], stoi=_stoi, context=N_CONTEXT)
X_DEV, Y_DEV = dataset(text=TEXT[N1:N2], stoi=_stoi, context=N_CONTEXT)
X_TEST, Y_TEST = dataset(text=TEXT[N2:], stoi=_stoi, context=N_CONTEXT)

# TRAIN ########################################################################

MODEL = Model()
# L_TRAIN, L_TEST, G_RATIOS = _mto.train(model=MODEL, loss=loss, x_train=X_TRAIN, y_train=Y_TRAIN, x_test=X_TEST, y_test=Y_TEST, steps=N_STEPS, batch=N_BATCH, rate=R_TRAINING)

# SAMPLING ####################################################################

sample = functools.partial(_ms.sample, model=MODEL, context=N_CONTEXT, depth=N_VOCABULARY, max_length=N_SAMPLE, itos=_itos)

# VIZ #########################################################################

# tf.argmax(X_TRAIN, axis=-1) # convert one_hot to indices and check the dataset
# tf.argmax(Y_TRAIN, axis=-1)

PATH = os.path.join('.logs/', VERSION, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
SUMMARY = tf.summary.create_file_writer(PATH)

# plot model stats
# _mts.save_model_histograms(model=MODEL, step=N_STEPS, summary=SUMMARY)

# plot loss
# _mts.save_loss_plot(data=L_TRAIN, name='train_loss', summary=SUMMARY, offset=0)
# _mts.save_loss_plot(data=L_TEST, name='test_loss', summary=SUMMARY, offset=0)

# plot log10(gradient / value)
# _mts.save_ratios_plot(data=G_RATIOS, model=MODEL, summary=SUMMARY, offset=0)
