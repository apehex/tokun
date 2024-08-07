"""Publish tokun to Hugging Face."""

import itertools
import math
import os
import tempfile
import urllib.request

import huggingface_hub as hh
import keras as ks
import tensorflow as tf
import transformers as ht

import mlable.io
import mlable.metrics

import tokun.evaluation
import tokun.huggingface
import tokun.meta
import tokun.model
import tokun.pipeline

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

BINARY = True

# META ########################################################################

N_SEQUENCE_AXIS = 1
N_FEATURE_AXIS = -1

N_TOKEN_DIM = [4, 16] # G, for each block
N_INPUT_DIM = 256 # U_i (bytes)
N_OUTPUT_DIM = 8 if BINARY else 256 # U_o (8 bits)
N_EMBEDDING_DIM = 256 # E
N_SEQUENCE_DIM = 512

OUTPUT = 'binary' if BINARY else 'categorical'

# DERIVED #####################################################################

N_TOKEN_SIZES = list(itertools.accumulate(N_TOKEN_DIM, lambda x, y: x * y)) # in bytes

VERSION = tokun.meta.version(token_units=N_TOKEN_DIM, sequence_axis=N_SEQUENCE_AXIS, input_dim=N_INPUT_DIM, embed_dim=N_EMBEDDING_DIM, output_dim=N_OUTPUT_DIM)
LABEL = '8.5'

PATH_IMPORT = os.path.join('models/', *VERSION, '{}.keras'.format(LABEL))
PATH_EXPORT = os.path.join('variants/', *VERSION[:2])

# TOKENIZER ###################################################################

TOKENIZER = tokun.huggingface.ByteTokenizer(vocab_size=256, split_special_tokens=True)

# METRICS #####################################################################

_Accuracy = mlable.metrics.BinaryGroupAccuracy if BINARY else mlable.metrics.CategoricalGroupAccuracy
_Loss = tf.keras.losses.BinaryCrossentropy if BINARY else tf.keras.losses.CategoricalCrossentropy

# COMPILE ########################################################################

with DISTRIBUTION_STRATEGY.scope():
    # metrics
    byte_accuracy = _Accuracy(group=1, name='byte_accuracy')
    character_accuracy = _Accuracy(group=4, name='character_accuracy')
    token_accuracy = _Accuracy(group=N_TOKEN_SIZES[-1], name='token_accuracy')
    # weights and config
    MODEL = tf.keras.models.load_model(PATH_IMPORT, compile=False)
    # compilation
    MODEL.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss=_Loss(from_logits=False, label_smoothing=0., axis=-1, reduction='sum_over_batch_size', name='loss'),
        metrics=[byte_accuracy, character_accuracy, token_accuracy])

# SPECIFY IO ##################################################################

__inputs = tf.keras.layers.Input(shape=(math.prod(N_TOKEN_DIM) * N_SEQUENCE_DIM,), dtype=tf.int32)

__outputs = MODEL._encoder(__inputs)
__outputs = MODEL._decoder(__outputs)

TOKUN = tf.keras.models.Model(__inputs, __outputs)

# SAMPLE ######################################################################

__s = """위키백과, 우리 모두의 백과사전.\nt-분포 확률적 임베딩(t-SNE)은 데이터의 차원 축소에 사용되는 기계 학습 알고리즘 중 하나로, 2002년 샘 로이스Sam Rowise와 제프리 힌튼에 의해 개발되었다.[1] t-SNE는 비선형 차원 축소 기법으로, 고차원 데이터를 특히 2, 3차원 등으로 줄여 가시화하는데에 유용하게 사용된다. 구체적으로 t-SNE는 비슷한 데이터는 근접한 2, 3차원의 지점으로, 다른 데이터는 멀리 떨어진 지점으로 맵핑한다."""

# UTF-32 TOKENIZATION #########################################################

__x = TOKENIZER.batch_encode_plus(batch_text_or_text_pairs=[__s], padding='max_length', max_length=math.prod(N_TOKEN_DIM) * N_SEQUENCE_DIM, add_special_tokens=False)
__x = tf.convert_to_tensor(__x['input_ids'])

# TEST THE DERIVED MODEL ######################################################

__e = TOKUN.layers[1](__x) # encoder
__p = TOKUN.layers[2](__e) # decoder
__y = tokun.pipeline.postprocess(__p, binary=BINARY, random=False)
__o = tokun.pipeline.unpack(__y)

# CHECK #######################################################################

print(MODEL.summary())
print(TOKUN.summary())

print(tokun.evaluation.compare(__s, __o[0]))
print(__s)
print(__o[0])

# INIT HF API #################################################################

API = hh.HfApi()

# TEMP ########################################################################

PATH_TEMP = tempfile.mkdtemp()
PATH_MODEL = os.path.join(PATH_TEMP, 'model/')
PATH_TEST = os.path.join(PATH_TEMP, 'test/')
PATH_TOKENIZER = os.path.join(PATH_TEMP, 'tokenizer/')

print(PATH_TEMP)

# TOKENIZER ###################################################################

TOKENIZER.save_pretrained(save_directory=PATH_TOKENIZER)
API.upload_folder(repo_id='apehex/tokun', folder_path=PATH_TOKENIZER, path_in_repo='tokenizer/')

# MODEL #######################################################################

hh.save_pretrained_keras(model=TOKUN, save_directory=PATH_MODEL, config=TOKUN.get_config())
API.upload_folder(repo_id='apehex/tokun', folder_path=PATH_MODEL, path_in_repo=PATH_EXPORT)

# TEST ########################################################################

API.snapshot_download(repo_id='apehex/tokun', local_dir=PATH_TEST)
__TOKENIZER = tokun.huggingface.ByteTokenizer()
__TOKUN = hh.from_pretrained_keras(os.path.join(PATH_TEST, PATH_EXPORT))

# PREDICT #####################################################################

__s = """위키백과, 우리 모두의 백과사전.\nt-분포 확률적 임베딩(t-SNE)은 데이터의 차원 축소에 사용되는 기계 학습 알고리즘 중 하나로, 2002년 샘 로이스Sam Rowise와 제프리 힌튼에 의해 개발되었다.[1] t-SNE는 비선형 차원 축소 기법으로, 고차원 데이터를 특히 2, 3차원 등으로 줄여 가시화하는데에 유용하게 사용된다. 구체적으로 t-SNE는 비슷한 데이터는 근접한 2, 3차원의 지점으로, 다른 데이터는 멀리 떨어진 지점으로 맵핑한다."""

__x = __TOKENIZER.batch_encode_plus(batch_text_or_text_pairs=[__s], padding='max_length', max_length=math.prod(N_TOKEN_DIM) * N_SEQUENCE_DIM, add_special_tokens=False)
__x = tf.convert_to_tensor(__x['input_ids'])

__p = __TOKUN(__x)
__y = tokun.pipeline.postprocess(__p, binary=BINARY, random=False)
__o = tokun.pipeline.unpack(__y)

print(tokun.evaluation.compare(__s, __o[0]))
print(__s)
print(__o[0])
