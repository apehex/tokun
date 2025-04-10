"""Apply the model to various samples."""

import itertools
import math
import os

import keras
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

# TOGGLE ######################################################################

BINARY = True

# META ########################################################################

TOKUN_CONFIG = {
    'token_dim': [4, 4, 4],
    'input_dim': 256,
    'embed_dim': 256,
    'output_dim': 8 if BINARY else 256,
    'sequence_axis': 1}

META_CONFIG = {
    'version': tokun.meta.version(**TOKUN_CONFIG),
    'label': '6.1',}

IO_CONFIG = {
    'path': os.path.join('models/', *META_CONFIG['version'], '{}.keras'.format(META_CONFIG['label'])),}

OPTIMIZER_CONFIG = {
    'learning_rate': 0.0001,
    'weight_decay': 0.1,
    'beta_1': 0.9,
    'beta_2': 0.99,
    'clipnorm': 1.0,}

LOSS_CONFIG = {
    'from_logits': False,
    'label_smoothing': 0.,
    'axis': -1,
    'reduction': 'sum_over_batch_size',
    'name': 'ce_loss',}

# METRICS #####################################################################

_Accuracy = mlable.metrics.BinaryGroupAccuracy if BINARY else mlable.metrics.CategoricalGroupAccuracy
_Loss = tf.keras.losses.BinaryCrossentropy if BINARY else tf.keras.losses.CategoricalCrossentropy

# INIT MODEL ##################################################################

with DISTRIBUTION_STRATEGY.scope():
    # metrics
    byte_accuracy = _Accuracy(group=1, name='byte_accuracy')
    character_accuracy = _Accuracy(group=4, name='character_accuracy')
    token_accuracy = _Accuracy(group=math.prod(TOKUN_CONFIG['token_dim']), name='token_accuracy')
    # weights and config
    MODEL = tf.keras.models.load_model(IO_CONFIG['path'], compile=False)
    # compilation
    MODEL.compile(
        optimizer=tf.keras.optimizers.Adam(**OPTIMIZER_CONFIG),
        loss=_Loss(**LOSS_CONFIG),
        metrics=[byte_accuracy, character_accuracy, token_accuracy])

# SAMPLES #####################################################################

SAMPLES = [
    """위키백과, 우리 모두의 백과사전.\nt-분포 확률적 임베딩(t-SNE)은 데이터의 차원 축소에 사용되는 기계 학습 알고리즘 중 하나로, 2002년 샘 로이스Sam Rowise와 제프리 힌튼에 의해 개발되었다.[1] t-SNE는 비선형 차원 축소 기법으로, 고차원 데이터를 특히 2, 3차원 등으로 줄여 가시화하는데에 유용하게 사용된다. 구체적으로 t-SNE는 비슷한 데이터는 근접한 2, 3차원의 지점으로, 다른 데이터는 멀리 떨어진 지점으로 맵핑한다.""",
    """class AutoEncoder(tf.keras.models.Model):\n    def __init__(self, token_dim: int, encoding_dim: int, embedding_dim: int, batch_dim: int=None, **kwargs) -> None:\n        super(AutoEncoder, self).__init__(**kwargs)\n        self._encoder = Encoder(token_dim=token_dim, encoding_dim=encoding_dim, embedding_dim=embedding_dim, batch_dim=batch_dim)\n        self._decoder = Decoder(token_dim=token_dim, encoding_dim=encoding_dim, embedding_dim=embedding_dim, batch_dim=batch_dim)\n\n    def call(self, x: tf.Tensor) -> tf.Tensor:\n        return self._decoder(self._encoder(x))""",
    """class AutoEncoder(tf.keras.models.Model):\n  def __init__(self, token_dim: int, encoding_dim: int, embedding_dim: int, batch_dim: int=None, **kwargs) -> None:\n    super(AutoEncoder, self).__init__(**kwargs)\n    self._encoder = Encoder(token_dim=token_dim, encoding_dim=encoding_dim, embedding_dim=embedding_dim, batch_dim=batch_dim)\n    self._decoder = Decoder(token_dim=token_dim, encoding_dim=encoding_dim, embedding_dim=embedding_dim, batch_dim=batch_dim)\n\n  def call(self, x: tf.Tensor) -> tf.Tensor:\n    return self._decoder(self._encoder(x))"""]

SAMPLES.extend([__i * chr(0) + SAMPLES[1] for __i in range(math.prod(TOKUN_CONFIG['token_dim']) // 4)])

# TEST ########################################################################

__x, __e, __p, __y, __o = tokun.pipeline.sample(model=MODEL, text=SAMPLES[0], token_size=math.prod(TOKUN_CONFIG['token_dim']), expand=[1], binary=BINARY, random=False)

print(tokun.evaluation.compare(SAMPLES[0], __o[0]))
print(SAMPLES[0])
print(__o[0])

# ROBUSTNESS ##################################################################

__std = tf.math.reduce_std(__e, axis=1)
__noise = tf.random.normal(shape=(TOKUN_CONFIG['embed_dim'],), mean=0., stddev=tf.math.reduce_mean(__std).numpy())

__x, __e, _, _, _ = tokun.pipeline.sample(model=MODEL, text='tokun to can tok', token_size=math.prod(TOKUN_CONFIG['token_dim']), expand=[1], binary=BINARY, random=False)

print(tokun.pipeline.unpack(tokun.pipeline.postprocess(MODEL._decoder(__e), binary=True, random=False)))
print(tokun.pipeline.unpack(tokun.pipeline.postprocess(MODEL._decoder(__e + 0.8 * __std), binary=True, random=False)))
print(tokun.pipeline.unpack(tokun.pipeline.postprocess(MODEL._decoder(__e + 0.4 * __noise), binary=True, random=False)))
