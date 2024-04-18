import tensorflow as tf

import mlable.tensorflow.layers as _mtl

# ENCODING BLOCKS #############################################################

class TokenizeBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        left_axis: int=-2,
        right_axis: int=-1,
        latent_dim: int=256,
        **kwargs
    ):
        super(TokenizeBlock, self).__init__(**kwargs)
        # layers
        self._embedding = _mtl.PositionalEmbedding(input_axis=left_axis, output_axis=right_axis, name='positional-embeddings') # (..., G, E) + (1, G, E)
        self._merge = _mtl.Merge(left_axis=left_axis, right_axis=right_axis, left=True, name='merged-embeddings') # (..., G, E) => (..., G * E)
        self._dense = tf.keras.layers.Dense(units=latent_dim, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', name='compressed-embeddings') # (..., G * E) => (..., L), typically L = E

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return self._dense(self._merge(self._embedding(inputs)))

# DECODING BLOCKS #############################################################
