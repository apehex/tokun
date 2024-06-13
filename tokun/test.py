"""Apply the model to various samples."""

import itertools
import os

import keras
import tensorflow as tf

import tokun.evaluation
import tokun.meta
import tokun.model
import tokun.pipeline

# META ########################################################################

N_SEQUENCE_AXIS = 1
N_TOKEN_DIM = [4, 4] # G, for each block

# DERIVED #####################################################################

N_TOKEN_SIZES = list(itertools.accumulate(N_TOKEN_DIM, lambda x, y: x * y)) # in bytes

# IMPORT ######################################################################

VERSION = tokun.meta.version(units=N_TOKEN_DIM, axis=N_SEQUENCE_AXIS)
LABEL = '8.5'

PATH_IMPORT = os.path.join('models/', *VERSION, '{}.keras'.format(LABEL))

MODEL = keras.models.load_model(PATH_IMPORT)

# SAMPLES #####################################################################

SAMPLES = [
    """위키백과, 우리 모두의 백과사전.\nt-분포 확률적 임베딩(t-SNE)은 데이터의 차원 축소에 사용되는 기계 학습 알고리즘 중 하나로, 2002년 샘 로이스Sam Rowise와 제프리 힌튼에 의해 개발되었다.[1] t-SNE는 비선형 차원 축소 기법으로, 고차원 데이터를 특히 2, 3차원 등으로 줄여 가시화하는데에 유용하게 사용된다. 구체적으로 t-SNE는 비슷한 데이터는 근접한 2, 3차원의 지점으로, 다른 데이터는 멀리 떨어진 지점으로 맵핑한다.""",
    """class AutoEncoder(tf.keras.models.Model):\n    def __init__(self, token_dim: int, encoding_dim: int, embedding_dim: int, latent_dim: int, batch_dim: int=None, **kwargs) -> None:\n        super(AutoEncoder, self).__init__(**kwargs)\n        self._encoder = Encoder(token_dim=token_dim, encoding_dim=encoding_dim, embedding_dim=embedding_dim, latent_dim=latent_dim, batch_dim=batch_dim)\n        self._decoder = Decoder(token_dim=token_dim, encoding_dim=encoding_dim, embedding_dim=embedding_dim, latent_dim=latent_dim, batch_dim=batch_dim)\n\n    def call(self, x: tf.Tensor) -> tf.Tensor:\n        return self._decoder(self._encoder(x))""",
    """class AutoEncoder(tf.keras.models.Model):\n  def __init__(self, token_dim: int, encoding_dim: int, embedding_dim: int, latent_dim: int, batch_dim: int=None, **kwargs) -> None:\n    super(AutoEncoder, self).__init__(**kwargs)\n    self._encoder = Encoder(token_dim=token_dim, encoding_dim=encoding_dim, embedding_dim=embedding_dim, latent_dim=latent_dim, batch_dim=batch_dim)\n    self._decoder = Decoder(token_dim=token_dim, encoding_dim=encoding_dim, embedding_dim=embedding_dim, latent_dim=latent_dim, batch_dim=batch_dim)\n\n  def call(self, x: tf.Tensor) -> tf.Tensor:\n    return self._decoder(self._encoder(x))"""]

SAMPLES.extend([__i * chr(0) + SAMPLES[1] for __i in range(N_TOKEN_SIZES[-1] // 4)])

# TEST ########################################################################

__x, __e, __p, __y = tokun.pipeline.sample(model=MODEL, text=SAMPLES[0], groups=N_TOKEN_DIM, expand=N_SEQUENCE_AXIS * [1], flatten=True)

print(SAMPLES[0])
print(__y)
print(tokun.evaluation.compare(SAMPLES[0], __y))

# ROBUSTNESS ##################################################################

__std = tf.math.reduce_std(__e, axis=0)
__noise = tf.random.normal(shape=(256,), mean=0., stddev=tf.math.reduce_mean(__std).numpy())

__x, __e, _, _ = tokun.pipeline.sample(model=MODEL, text='tokun to can tok', groups=N_TOKEN_DIM, expand=N_SEQUENCE_AXIS * [1], flatten=True)

print(tokun.pipeline.postprocess(MODEL._decoder(__e)))
print(tokun.pipeline.postprocess(MODEL._decoder(__e + 1.6 * __std)))
print(tokun.pipeline.postprocess(MODEL._decoder(__e + 0.8 * __noise)))
