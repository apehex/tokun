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
import mlable.tensorflow.models as _mtm
import mlable.tensorflow.optimizers as _mto
import mlable.tensorflow.summary as _mts

# META ########################################################################

N_VOCABULARY = 37
N_CONTEXT = 16
N_EMBEDDING = 64
N_ATTENTION = 64
N_HIDDEN = 4 * N_ATTENTION

N_EPOCHS = 2
N_BATCH = 128

N_SAMPLE = 256

R_TRAINING = 0.1

VERSION = 'sat-tf-125k'

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

class Model(tf.Module):
    def __init__(self, n_context: int=N_CONTEXT, n_vocabulary: int=N_VOCABULARY, n_embedding: int=N_EMBEDDING, n_attention: int=N_ATTENTION, n_hidden: int=N_HIDDEN, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # layers
        self._layers = [
            # embedding
            _mtl.Embedding(input_dim=n_vocabulary, output_dim=n_embedding, add_position=True, name='embedding'),
            # block 1
            _mtm.ResidualSelfAttentionBlock(attention_head_dim=n_attention, attention_head_count=1, name='attention-block-1'),
            # block 2
            _mtm.FeedForwardResidualBlock(hidden_dim=n_hidden, name='feedforward-block-1'),
            # because of the residual connections, all internal layers have shape (-1, n_context, n_embedding) = (-1, n_context, n_attention)
            _mtl.Reshape(target_shape=(-1, n_context * n_embedding)),
            # head
            _mtl.Dense(units=n_vocabulary, use_bias=True, name='head'),
            _mtl.Softmax(axis=-1, name='softmax')]

    def __call__(self, x, training: bool=True):
        __x = x
        # propagate x
        for __l in self._layers:
            __x = __l(__x, training=training)
        # return the output of the latest layer
        return tf.squeeze(__x)

    def n_trainable_elements(self):
        return tf.reduce_sum([tf.size(__v) for __v in MODEL.trainable_variables]).numpy()

# LOSS ########################################################################

loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False, label_smoothing=0., axis=-1, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE, name='loss')

# TEST ########################################################################

# DATA ########################################################################

N1 = int(0.8 * len(TEXT))
N2 = int(0.9 * len(TEXT))

X_TRAIN, Y_TRAIN = _min.dataset(text=TEXT[:N1], stoi=_stoi, context=N_CONTEXT, depth=N_VOCABULARY)
X_DEV, Y_DEV = _min.dataset(text=TEXT[N1:N2], stoi=_stoi, context=N_CONTEXT, depth=N_VOCABULARY)
X_TEST, Y_TEST = _min.dataset(text=TEXT[N2:], stoi=_stoi, context=N_CONTEXT, depth=N_VOCABULARY)

# TRAIN ########################################################################

MODEL = Model()
# L_TRAIN, L_TEST, G_RATIOS = _mto.train(model=MODEL, loss=loss, x_train=X_TRAIN, y_train=Y_TRAIN, x_test=X_TEST, y_test=Y_TEST, epochs=N_EPOCHS, batch=N_BATCH, rate=R_TRAINING)

# SAMPLING ####################################################################

sample = functools.partial(_ms.sample, model=MODEL, context=N_CONTEXT, depth=N_VOCABULARY, length=N_SAMPLE, itos=_itos)

# VIZ #########################################################################

# tf.argmax(X_TRAIN, axis=-1) # convert one_hot to indices and check the dataset
# tf.argmax(Y_TRAIN, axis=-1)

PATH = os.path.join('.logs/', VERSION, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
SUMMARY = tf.summary.create_file_writer(PATH)

# plot model stats
# _mts.save_model_histograms(model=MODEL, epoch=N_EPOCHS, summary=SUMMARY)

# plot loss
# _mts.save_loss_plot(data=L_TRAIN, name='train_loss', summary=SUMMARY, offset=0)
# _mts.save_loss_plot(data=L_TEST, name='test_loss', summary=SUMMARY, offset=0)

# plot log10(gradient / value)
# _mts.save_ratios_plot(data=G_RATIOS, model=MODEL, summary=SUMMARY, offset=0)
