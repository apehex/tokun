"""Train tokun from scratch or from a checkpoint."""

import datetime
import functools
import itertools
import math
import os

import tensorflow as tf
import tensorflow_datasets as tfds

import mlable.data
import mlable.metrics

import tokun.data
import tokun.meta
import tokun.model
import tokun.pipeline

# MIXED PRECISION #############################################################

tf.keras.mixed_precision.set_global_policy('mixed_float16') # mixed_bfloat16 on TPUs

# DEVICES #####################################################################

tf.debugging.set_log_device_placement(False)

CPU = tf.config.list_logical_devices('CPU')
GPU = tf.config.list_logical_devices('GPU')
TPU = tf.config.list_logical_devices('TPU')

if TPU:
    RESOLVER = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(RESOLVER)
    tf.tpu.experimental.initialize_tpu_system(RESOLVER)
    DISTRIBUTION_STRATEGY = tf.distribute.TPUStrategy(RESOLVER)
elif GPU:
    DISTRIBUTION_STRATEGY = tf.distribute.MirroredStrategy(GPU)
else:
    DISTRIBUTION_STRATEGY = tf.distribute.MirroredStrategy(CPU)

print(DISTRIBUTION_STRATEGY)

# TOGGLE ######################################################################

IMPORT = False
TRAINING = True
RANDOM = True
BINARY = True

# MODEL PARAMETERS ############################################################

N_SEQUENCE_AXIS = 1
N_FEATURE_AXIS = -1

N_TOKEN_DIM = [4, 4, 4] # G, for each block
N_INPUT_DIM = 256 # U_i (bytes)
N_OUTPUT_DIM = 8 if BINARY else 256 # U_o (8 bits)
N_EMBEDDING_DIM = 256 # E

OUTPUT = 'binary' if BINARY else 'categorical'

# TRAINING PARAMETERS #########################################################

N_EPOCHS = 8

N_BATCH_DIM = 128 # number of samples per batch
N_SAMPLE_DIM = 256 # number of characters per sample (=> 4 * N_SAMPLE_DIM integers per sample)

R_0, B_0, B_1, B_2 = tokun.meta.rates(pretrained=IMPORT, normalization=True, base=0.001)

CLASS_WEIGHTS = {__c: 0.3 if __c == 0 else 1. for __c in range(N_INPUT_DIM)} # there are 3 times more 0s than other bytes

# DERIVED #####################################################################

N_TOKEN_SIZES = list(itertools.accumulate(N_TOKEN_DIM, lambda x, y: x * y)) # in bytes
N_OFFSET_TICKS = [2 ** __i for __i in range(int(math.log(N_TOKEN_SIZES[-1] // 4, 2)))] # in characters

# IMPORT ######################################################################

PATH_IMPORT = os.path.join('models/256x8/4x16/1/8.5.keras')

# LOG #########################################################################

VERSION = tokun.meta.version(token_units=N_TOKEN_DIM, sequence_axis=N_SEQUENCE_AXIS, input_dim=N_INPUT_DIM, embed_dim=N_EMBEDDING_DIM, output_dim=N_OUTPUT_DIM)
DATETIME = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

PATH_LOG = os.path.join('.logs/', *VERSION, DATETIME)
PATH_EXPORT = os.path.join('models/', *VERSION, DATETIME + '.keras')

# DATA ########################################################################

LANG = ['ar', 'de', 'en', 'es', 'hi', 'vi', 'zh']
MLQA_TRAIN = {__l: tfds.load('mlqa/' + __l, split='test', as_supervised=False, shuffle_files=True, data_dir='~/.cache/tensorflow/', batch_size=None) for __l in LANG}
MLQA_TEST = {__l: tfds.load('mlqa/' + __l, split='validation', as_supervised=False, shuffle_files=True, data_dir='~/.cache/tensorflow/', batch_size=None) for __l in LANG}

RANDOM_TRAIN = tokun.data.random_dataset(size=N_BATCH_DIM * 2 ** 14, sample_size=N_SAMPLE_DIM, lower_plane=0, upper_plane=0x40000)
RANDOM_TEST = tokun.data.random_dataset(size=N_BATCH_DIM * 2 ** 8, sample_size=N_SAMPLE_DIM, lower_plane=0, upper_plane=0x40000)

# OUTPUT ENCODING #############################################################

_encode_binary = lambda __x: tf.cast(mlable.ops.expand_base(__x, base=2, depth=N_OUTPUT_DIM), dtype=tf.dtypes.float32)
_encode_categorical = lambda __x: tf.one_hot(__x, depth=N_OUTPUT_DIM, axis=-1)
_encode_output = _encode_binary if BINARY else _encode_categorical

# PREPROCESS MLQA #############################################################

PIPELINE = [
    # join the features
    ((lambda __x: tf.strings.join(inputs=[__x['context'], __x['question']], separator='\x1d')), True),
    # offset by 1 to 15 character => (1,) bytes
    *[(functools.partial(tokun.pipeline.offset, ticks=__t), False) for __t in N_OFFSET_TICKS], # (offsets 0, ..., (2 ^ i) - 1) + (offsets 2 ^ i, ..., 2 ^ (i+1) - 1)
    # encode => (B, 4 * S,) int
    (functools.partial(tokun.pipeline.encode, token_size=N_TOKEN_SIZES[-1], sample_size=N_SAMPLE_DIM), True),
    # reshape => (B, 4 * S,) int
    (functools.partial(tf.reshape, shape=(N_BATCH_DIM, 4 * N_SAMPLE_DIM,)), True),
    # encode classes on 8 bits for the 256 possibilities / byte
    ((lambda __x: (__x, _encode_output(__x))), True)]

OPERATIONS, REPLACE = zip(*PIPELINE)

MLQA_TRAIN = {__l: mlable.data.process(dataset=__d.batch(N_BATCH_DIM, drop_remainder=True, num_parallel_calls=tf.data.AUTOTUNE), pipeline=OPERATIONS, replace=REPLACE) for __l, __d in MLQA_TRAIN.items()}
MLQA_TEST = {__l: mlable.data.process(dataset=__d.batch(N_BATCH_DIM, drop_remainder=True, num_parallel_calls=tf.data.AUTOTUNE), pipeline=OPERATIONS, replace=REPLACE) for __l, __d in MLQA_TEST.items()}

# PREPROCESS RANDOM ###########################################################

PIPELINE = [
    # reshape => (B, 4 * S,) int
    (functools.partial(tf.reshape, shape=(N_BATCH_DIM, 4 * N_SAMPLE_DIM,)), True),
    # encode classes on 8 bits for the 256 possibilities / byte
    ((lambda __x: (__x, _encode_output(__x))), True)]

OPERATIONS, REPLACE = zip(*PIPELINE)

RANDOM_TRAIN = mlable.data.process(dataset=RANDOM_TRAIN.batch(N_BATCH_DIM, drop_remainder=True, num_parallel_calls=tf.data.AUTOTUNE), pipeline=OPERATIONS, replace=REPLACE)
RANDOM_TEST = mlable.data.process(dataset=RANDOM_TEST.batch(N_BATCH_DIM, drop_remainder=True, num_parallel_calls=tf.data.AUTOTUNE), pipeline=OPERATIONS, replace=REPLACE)

# COMBINE DATASETS ############################################################

MLQA_TRAIN_ALL = functools.reduce(lambda __l, __r: __l.concatenate(__r), MLQA_TRAIN.values())
MLQA_TEST_ALL = functools.reduce(lambda __l, __r: __l.concatenate(__r), MLQA_TEST.values())

DATASET_TRAIN = RANDOM_TRAIN if RANDOM else MLQA_TRAIN_ALL
DATASET_TEST = MLQA_TEST_ALL

# INSPECT #####################################################################

print(RANDOM_TRAIN.element_spec)
print(RANDOM_TEST.element_spec)

print(DATASET_TRAIN.element_spec)
print(DATASET_TEST.element_spec)

print('train: {:,}'.format(DATASET_TRAIN.cardinality().numpy()))
print('test:  {:,}'.format(DATASET_TEST.cardinality().numpy()))

# METRICS #####################################################################

_Accuracy = mlable.metrics.BinaryGroupAccuracy if BINARY else mlable.metrics.CategoricalGroupAccuracy
_Loss = tf.keras.losses.BinaryCrossentropy if BINARY else tf.keras.losses.CategoricalCrossentropy

# COMPILE ########################################################################

with DISTRIBUTION_STRATEGY.scope():
    # metrics
    byte_accuracy = _Accuracy(group=1, name='byte_accuracy')
    character_accuracy = _Accuracy(group=4, name='character_accuracy')
    token_accuracy = _Accuracy(group=N_TOKEN_SIZES[-1], name='token_accuracy')
    # weights
    MODEL = tokun.model.AutoEncoder(sequence_axis=N_SEQUENCE_AXIS, feature_axis=N_FEATURE_AXIS, token_dim=N_TOKEN_DIM, input_dim=N_INPUT_DIM, output_dim=N_OUTPUT_DIM, embedding_dim=N_EMBEDDING_DIM, activation='gelu', output=OUTPUT)
    if IMPORT and os.path.isfile(PATH_IMPORT): MODEL = tf.keras.models.load_model(PATH_IMPORT, compile=False)
    # compile
    MODEL.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=R_0, weight_decay=B_0, beta_1=B_1, beta_2=B_2, clipnorm=1.0),
        loss=_Loss(from_logits=False, label_smoothing=0., axis=-1, reduction='sum_over_batch_size', name='ce_loss'),
        metrics=[byte_accuracy, character_accuracy, token_accuracy])

# TRAIN #######################################################################

if TRAINING:
    with DISTRIBUTION_STRATEGY.scope():
        # callbacks
        cp_callback = tf.keras.callbacks.ModelCheckpoint(PATH_EXPORT, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', save_freq='epoch')
        tb_callback = tf.keras.callbacks.TensorBoard(log_dir=PATH_LOG, histogram_freq=1, embeddings_freq=0, profile_batch=(16, 32), write_graph=False, write_images=True)
        # fit model
        TRAINING_HISTORY = MODEL.fit(
            x=DATASET_TRAIN.prefetch(tf.data.AUTOTUNE),
            batch_size=None,
            epochs=N_EPOCHS,
            validation_split=None,
            validation_data=DATASET_TEST.prefetch(tf.data.AUTOTUNE),
            validation_freq=list(range(1, N_EPOCHS + 1, 1)),
            class_weight=CLASS_WEIGHTS,
            verbose=1,
            callbacks=[cp_callback, tb_callback])
