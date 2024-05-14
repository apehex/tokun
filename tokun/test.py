"""Apply the model to various samples."""

import datetime
import functools
import math
import os

import keras
import tensorflow as tf

import tokun.meta
import tokun.model
import tokun.pipeline

# META ########################################################################

ATTENTION = True
NORMALIZATION = True

N_TOKEN_DIM = [4, 4, 4] # G, for each block

N_BATCH = 128 # number of samples per batch
N_SAMPLE = 128 # number of characters per sample (=> N_TOKEN_DIM * N_SAMPLE integers per sample)

# LOG #########################################################################

VERSION = tokun.meta.version(groups=N_TOKEN_DIM, attention=ATTENTION, normalization=NORMALIZATION)
DATETIME = '20240509-211600'

PATH_IMPORT = os.path.join('models/', *VERSION, DATETIME + '.keras')

# LOAD ########################################################################

MODEL = keras.models.load_model(PATH_IMPORT)

# SAMPLES ########################################################################

# TEST ########################################################################

__s = """class Encoder(tf.keras.models.Model):\n    def __init__(self, depth: int, token_dim: int, encoding_dim: int, embedding_dim: int, latent_dim: int, batch_dim: int=None, attention: bool=False, **kwargs) -> None:\n        super(Encoder, self).__init__(**kwargs)\n        self._encoder = tf.keras.Sequential([\n            tf.keras.Input(shape=(encoding_dim,), batch_size=batch_dim, name='input'), # (B * G ^ D, U)\n            tf.keras.layers.Dense(units=embedding_dim, activation=None, use_bias=False, kernel_initializer='glorot_uniform', bias_initializer=None, name='embed-1'),] # (B * G ^ D, U) => (B * G ^ D, E)\n            + [tokun.layers.TokenizeBlock(left_axis=-2, right_axis=-1, token_dim=token_dim, latent_dim=latent_dim, attention=attention, name='tokenize' + (__i + 1) * '-4') for __i in range(depth)]) # (B * G ^ i, E) => (B * G ^ (i-1), E)\n\n    def call(self, x: tf.Tensor) -> tf.Tensor:\n        return self._encoder(x)\n"""

__x = tokun.pipeline.preprocess(text=__s, groups=N_TOKEN_DIM, flatten=True)
__e = MODEL._encoder(__x)
__p = MODEL(__x)
__y = tokun.pipeline.postprocess(__p)

print(__s)
print(__y)
print(tokun.pipeline.compare(__s, __y))

# SHIFT #######################################################################

__sample  = """The t-SNE algorithm comprises two main stages. First, t-SNE constructs a probability distribution over pairs of high-dimensional objects in such a way that similar objects are assigned a higher probability while dissimilar points are assigned a lower probability. Second, t-SNE defines a similar probability distribution over the points in the low-dimensional map, and it minimizes the Kullback–Leibler divergence (KL divergence) between the two distributions with respect to the locations of the points in the map. While the original algorithm uses the Euclidean distance between objects as the base of its similarity metric, this can be changed as appropriate. A Riemannian variant is UMAP.\n\nt-SNE has been used for visualization in a wide range of applications, including genomics, computer security research,[3] natural language processing, music analysis,[4] cancer research,[5] bioinformatics,[6] geological domain interpretation,[7][8][9] and biomedical signal processing.[10]\n\nWhile t-SNE plots often seem to display clusters, the visual clusters can be influenced strongly by the chosen parameterization and therefore a good understanding of the parameters for t-SNE is necessary. Such "clusters" can be shown to even appear in non-clustered data,[11] and thus may be false findings. Interactive exploration may thus be necessary to choose parameters and validate results.[12][13] It has been demonstrated that t-SNE is often able to recover well-separated clusters, and with special parameter choices, approximates a simple form of spectral clustering.[14]"""

# compute
__s = ''.join(__i * chr(0) + __sample for __i in range(4))
__t = tokun.pipeline.chunk(sequence=__s, size=4, repeats=False)
__x = tokun.pipeline.preprocess(text=''.join(__t), groups=N_TOKEN_DIM, flatten=True)
__e = MODEL._encoder(__x)
__p = MODEL(__x)
__y = tokun.pipeline.postprocess(__p)

# print
print('# INPUT ################################################################\n\n' + ''.join(__t))
print('\n# OUTPUT ###############################################################\n\n' + __y)
print('\n# SCORE ################################################################\n\n' + str(tokun.pipeline.compare(''.join(__t), __y)))
print('\n# SHAPES ###############################################################\n')
print(len(__t))
print(__x.shape)
print(__e.shape)
print(__p.shape)

# save
write(data=[__c + ' ' + label(__c) for __c in __t], path='./metadata.4.shift.tsv', tsv=False)
write(data=__e.numpy(), path='./embeddings.4.shift.tsv', tsv=True)
