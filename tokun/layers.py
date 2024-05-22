"""Encoding and decoding blocks of tokun."""

import keras
import tensorflow as tf

import mlable.tensorflow.layers as _mtl

# ENCODING BLOCKS #############################################################

@keras.saving.register_keras_serializable(package='blocks')
class TokenizeBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        left_axis: int=-2,
        right_axis: int=-1,
        token_dim: int=4,
        latent_dim: int=256,
        attention: bool=False,
        normalization: bool=False,
        activation: str='silu',
        **kwargs
    ) -> None:
        super(TokenizeBlock, self).__init__(**kwargs)
        # layers
        self._normalization = tf.keras.layers.LayerNormalization(axis=-1, epsilon=0.001, center=True, scale=True, name='normalization') if normalization else None # normalize each token unit independently
        self._divide = _mtl.Divide(input_axis=0, output_axis=1, factor=token_dim, insert=True, name='group') # (B * G, E) => (B, G, E)
        self._embedding = _mtl.PositionalEmbedding(input_axis=left_axis, output_axis=right_axis, name='position') # (B, G, E) + (1, G, E)
        self._attention = tf.keras.layers.Attention(use_scale=False, score_mode='dot', dropout=0., seed=None, name='attention') if attention else None # (B, G, E) + (B, G, E) * (B, E, G) * (B, G, E)
        self._merge = _mtl.Merge(left_axis=left_axis, right_axis=right_axis, left=True, name='merging') # (B, G, E) => (B, G * E)
        self._dense = tf.keras.layers.Dense(units=latent_dim, activation=activation, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', name='compression') # (B, G * E) => (B, L), typically L = E

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        __t = self._normalization(inputs) if self._normalization else inputs
        __t = self._embedding(self._divide(__t))
        __t = self._attention([__t, __t, __t], return_attention_scores=False, use_causal_mask=False) if self._attention else __t
        return self._dense(self._merge(__t))

    def get_config(self) -> dict:
        __parent_config = super(TokenizeBlock, self).get_config()
        __child_config = {
            'left_axis': self._merge._left_axis,
            'right_axis': self._merge._right_axis,
            'token_dim': self._divide._factor,
            'latent_dim': self._dense.units,
            'attention': bool(self._attention),
            'normalization': bool(self._normalization),
            'activation': self._dense.activation.__name__,}
        return {**__parent_config, **__child_config}

    @classmethod
    def from_config(cls, config) -> tf.keras.layers.Layer:
        return cls(**config)

# DECODING BLOCKS #############################################################

@keras.saving.register_keras_serializable(package='blocks')
class DetokenizeBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        token_dim: int=4,
        embedding_dim: int=256,
        attention: bool=False,
        normalization: bool=False,
        activation: str='silu',
        **kwargs
    ) -> None:
        super(DetokenizeBlock, self).__init__(**kwargs)
        # layers
        self._dense = tf.keras.layers.Dense(units=token_dim * embedding_dim, activation=activation, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', name='decompression') # (B, L) => (B, G * E), typically L = E
        self._divide = _mtl.Divide(input_axis=-2, output_axis=-1, insert=True, factor=embedding_dim, name='split') # (B, G * E) => (B, G, E)
        self._embedding = _mtl.PositionalEmbedding(input_axis=-2, output_axis=-1, name='position') # (B, G, E) + (1, G, E)
        self._attention = tf.keras.layers.Attention(use_scale=False, score_mode='dot', dropout=0., seed=None, name='attention') if attention else None # (B, G, E) + (B, G, E) * (B, E, G) * (B, G, E)
        self._merge = _mtl.Merge(left_axis=0, right_axis=1, left=True) # (B, G, E) => (B * G, E)
        self._normalization = tf.keras.layers.LayerNormalization(axis=-1, epsilon=0.001, center=True, scale=True, name='normalization') if normalization else None # normalize each token unit independently

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        __t = self._embedding(self._divide(self._dense(inputs)))
        __t = self._attention([__t, __t, __t], return_attention_scores=False, use_causal_mask=False) if self._attention else __t
        __t = self._merge(__t)
        return self._normalization(__t) if self._normalization else __t

    def get_config(self) -> dict:
        __parent_config = super(DetokenizeBlock, self).get_config()
        __child_config = {
            'token_dim': self._dense.units // self._divide._factor,
            'embedding_dim': self._divide._factor,
            'attention': bool(self._attention),
            'normalization': bool(self._normalization),
            'activation': self._dense.activation.__name__,}
        return {**__parent_config, **__child_config}

    @classmethod
    def from_config(cls, config) -> tf.keras.layers.Layer:
        return cls(**config)

# HEAD ########################################################################

@keras.saving.register_keras_serializable(package='blocks')
class HeadBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        encoding_dim: int=256,
        **kwargs
    ) -> None:
        super(HeadBlock, self).__init__(**kwargs)
        # layers
        self._dense = tf.keras.layers.Dense(units=encoding_dim, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', name='project-head') # (..., G, E) => (..., G, U), typically U = E
        self._softmax = tf.keras.layers.Softmax(axis=-1, name='softmax') # (..., G, U)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return self._softmax(self._dense(inputs))

    def get_config(self) -> dict:
        __parent_config = super(HeadBlock, self).get_config()
        __child_config = {'encoding_dim': self._dense.units,}
        return {**__parent_config, **__child_config}

    @classmethod
    def from_config(cls, config) -> tf.keras.layers.Layer:
        return cls(**config)
