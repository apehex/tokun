"""Apply the model to various samples."""

# SETUP ENV ###################################################################

import os

os.environ['KERAS_BACKEND'] = 'tensorflow'

# LOAD DEPS ###################################################################

import itertools
import math

import keras as ks
import tensorflow as tf

import mlable.metrics

import tokun.evaluation
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

# META ########################################################################

N_SEQUENCE_AXIS = 1
N_TOKEN_DIM = [4, 16] # G, for each block

# DERIVED #####################################################################

N_TOKEN_SIZES = list(itertools.accumulate(N_TOKEN_DIM, lambda x, y: x * y)) # in bytes

# IMPORT MODEL ################################################################

VERSION = tokun.meta.version(units=N_TOKEN_DIM, axis=N_SEQUENCE_AXIS)
LABEL = '7.3'

PATH_IMPORT = os.path.join('models/', *VERSION, '{}.keras'.format(LABEL))

# INIT MODEL ##################################################################

with DISTRIBUTION_STRATEGY.scope():
    # metrics
    byte_accuracy = mlable.metrics.CategoricalGroupAccuracy(group=1, name='byte_accuracy')
    character_accuracy = mlable.metrics.CategoricalGroupAccuracy(group=4, name='character_accuracy')
    token_accuracy = mlable.metrics.CategoricalGroupAccuracy(group=N_TOKEN_SIZES[-1], name='token_accuracy')
    # weights and config
    MODEL = ks.models.load_model(PATH_IMPORT, compile=False)
    # compilation
    MODEL.compile(
        optimizer=ks.optimizers.Adam(learning_rate=0.0001),
        loss=ks.losses.CategoricalCrossentropy(from_logits=False, label_smoothing=0., axis=-1, reduction='sum_over_batch_size', name='cce_loss'),
        metrics=[byte_accuracy, character_accuracy, token_accuracy])

# SAMPLES #####################################################################

SAMPLES = [
    """위키백과, 우리 모두의 백과사전.\nt-분포 확률적 임베딩(t-SNE)은 데이터의 차원 축소에 사용되는 기계 학습 알고리즘 중 하나로, 2002년 샘 로이스Sam Rowise와 제프리 힌튼에 의해 개발되었다.[1] t-SNE는 비선형 차원 축소 기법으로, 고차원 데이터를 특히 2, 3차원 등으로 줄여 가시화하는데에 유용하게 사용된다. 구체적으로 t-SNE는 비슷한 데이터는 근접한 2, 3차원의 지점으로, 다른 데이터는 멀리 떨어진 지점으로 맵핑한다.""",
    """class AutoEncoder(ks.models.Model):\n    def __init__(self, token_dim: int, encoding_dim: int, embedding_dim: int, batch_dim: int=None, **kwargs) -> None:\n        super(AutoEncoder, self).__init__(**kwargs)\n        self._encoder = Encoder(token_dim=token_dim, encoding_dim=encoding_dim, embedding_dim=embedding_dim, batch_dim=batch_dim)\n        self._decoder = Decoder(token_dim=token_dim, encoding_dim=encoding_dim, embedding_dim=embedding_dim, batch_dim=batch_dim)\n\n    def call(self, x: tf.Tensor) -> tf.Tensor:\n        return self._decoder(self._encoder(x))""",
    """class AutoEncoder(ks.models.Model):\n  def __init__(self, token_dim: int, encoding_dim: int, embedding_dim: int, batch_dim: int=None, **kwargs) -> None:\n    super(AutoEncoder, self).__init__(**kwargs)\n    self._encoder = Encoder(token_dim=token_dim, encoding_dim=encoding_dim, embedding_dim=embedding_dim, batch_dim=batch_dim)\n    self._decoder = Decoder(token_dim=token_dim, encoding_dim=encoding_dim, embedding_dim=embedding_dim, batch_dim=batch_dim)\n\n  def call(self, x: tf.Tensor) -> tf.Tensor:\n    return self._decoder(self._encoder(x))"""]

SAMPLES.extend([__i * chr(0) + SAMPLES[1] for __i in range(N_TOKEN_SIZES[-1] // 4)])

# TEST ########################################################################

__x, __e, __p, __y = tokun.pipeline.sample(model=MODEL, text=SAMPLES[0], token_size=math.prod(N_TOKEN_DIM), expand=N_SEQUENCE_AXIS * [1])

print(SAMPLES[0])
print(__y)
print(tokun.evaluation.compare(SAMPLES[0], __y))

# ROBUSTNESS ##################################################################

__std = ks.ops.std(__e, axis=1)
__noise = ks.random.normal(shape=(256,), mean=0., stddev=ks.ops.mean(__std).numpy())

__x, __e, _, _ = tokun.pipeline.sample(model=MODEL, text='tokun to can tok', token_size=math.prod(N_TOKEN_DIM), expand=N_SEQUENCE_AXIS * [1])

print(tokun.pipeline.postprocess(MODEL._decoder(__e)))
print(tokun.pipeline.postprocess(MODEL._decoder(__e + 1.6 * __std)))
print(tokun.pipeline.postprocess(MODEL._decoder(__e + 0.8 * __noise)))
