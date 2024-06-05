"""Encoding and decoding blocks of tokun."""

import keras
import tensorflow as tf

import mlable.layers.embedding
import mlable.layers.reshaping

# ENCODING BLOCKS #############################################################

@keras.saving.register_keras_serializable(package='blocks')
class TokenizeBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        feature_axis: int=-1,
        token_dim: int=4,
        latent_dim: int=256,
        attention: bool=False,
        normalization: bool=False,
        activation: str='silu',
        **kwargs
    ) -> None:
        super(TokenizeBlock, self).__init__(**kwargs)
        # this axis is inserted and then merged
        __temp_axis = 1
        # config
        self._config = {
            'feature_axis': feature_axis,
            'token_dim': token_dim,
            'latent_dim': latent_dim,
            'attention': attention,
            'normalization': normalization,
            'activation': activation,}
        # layers
        self._normalization = tf.keras.layers.LayerNormalization(axis=feature_axis, epsilon=0.001, center=True, scale=True, name='normalization') if normalization else None # normalize each token unit independently
        self._divide = mlable.layers.reshaping.Divide(input_axis=0, output_axis=__temp_axis, factor=token_dim, insert=True, name='group') # (B * G, E) => (B, G, E)
        self._embedding = mlable.layers.embedding.PositionalEmbedding(input_axis=__temp_axis, output_axis=feature_axis, name='position') # (B, G, E) + (1, G, E)
        self._attention = tf.keras.layers.MultiHeadAttention(num_heads=1, key_dim=latent_dim, value_dim=latent_dim, attention_axes=[__temp_axis], name='attention') if attention else None # (B, G, E) + (B, G, E) * (B, E, G) * (B, G, E)
        self._merge = mlable.layers.reshaping.Merge(left_axis=__temp_axis, right_axis=feature_axis, left=False, name='merging') # (B, G, E) => (B, G * E)
        self._dense = tf.keras.layers.Dense(units=latent_dim, activation=activation, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', name='compression') # (B, G * E) => (B, L), typically L = E

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        __t = self._normalization(inputs) if self._normalization else inputs
        __t = self._embedding(self._divide(__t))
        __t = self._attention(query=__t, key=__t, value=__t, return_attention_scores=False, use_causal_mask=False) if self._attention else __t
        return self._dense(self._merge(__t))

    def get_config(self) -> dict:
        __config = super(TokenizeBlock, self).get_config()
        __config.update(self._config)
        return __config

    @classmethod
    def from_config(cls, config) -> tf.keras.layers.Layer:
        return cls(**config)

# DECODING BLOCKS #############################################################

@keras.saving.register_keras_serializable(package='blocks')
class DetokenizeBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        feature_axis: int=-1,
        token_dim: int=4,
        embedding_dim: int=256,
        attention: bool=False,
        normalization: bool=False,
        activation: str='silu',
        **kwargs
    ) -> None:
        super(DetokenizeBlock, self).__init__(**kwargs)
        # this axis is inserted and then merged
        __temp_axis = 1
        # config
        self._config = {
            'feature_axis': feature_axis,
            'token_dim': token_dim,
            'embedding_dim': embedding_dim,
            'attention': attention,
            'normalization': normalization,
            'activation': activation,}
        # layers
        self._dense = tf.keras.layers.Dense(units=token_dim * embedding_dim, activation=activation, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', name='decompression') # (B, L) => (B, G * E), typically L = E
        self._divide = mlable.layers.reshaping.Divide(input_axis=feature_axis, output_axis=__temp_axis, insert=True, factor=token_dim, name='split') # (B, G * E) => (B, G, E)
        self._embedding = mlable.layers.embedding.PositionalEmbedding(input_axis=__temp_axis, output_axis=feature_axis, name='position') # (B, G, E) + (1, G, E)
        self._attention = tf.keras.layers.MultiHeadAttention(num_heads=1, key_dim=embedding_dim, value_dim=embedding_dim, attention_axes=[__temp_axis], name='attention') if attention else None  # (B, G, E) + (B, G, E) * (B, E, G) * (B, G, E)
        self._merge = mlable.layers.reshaping.Merge(left_axis=0, right_axis=__temp_axis, left=True) # (B, G, E) => (B * G, E)
        self._normalization = tf.keras.layers.LayerNormalization(axis=feature_axis, epsilon=0.001, center=True, scale=True, name='normalization') if normalization else None # normalize each token unit independently

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        __t = self._embedding(self._divide(self._dense(inputs)))
        __t = self._attention(query=__t, key=__t, value=__t, return_attention_scores=False, use_causal_mask=False) if self._attention else __t
        __t = self._merge(__t)
        return self._normalization(__t) if self._normalization else __t

    def get_config(self) -> dict:
        __config = super(DetokenizeBlock, self).get_config()
        __config.update(self._config)
        return __config

    @classmethod
    def from_config(cls, config) -> tf.keras.layers.Layer:
        return cls(**config)

# HEAD ########################################################################

@keras.saving.register_keras_serializable(package='blocks')
class HeadBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        feature_axis: int=-1,
        encoding_dim: int=256,
        **kwargs
    ) -> None:
        super(HeadBlock, self).__init__(**kwargs)
        # config
        self._config = {'feature_axis': feature_axis, 'encoding_dim': encoding_dim}
        # layers
        self._dense = tf.keras.layers.Dense(units=encoding_dim, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', name='projection') # (..., G, E) => (..., G, U), typically U = E
        self._softmax = tf.keras.layers.Softmax(axis=feature_axis, name='softmax') # (..., G, U)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return self._softmax(self._dense(inputs))

    def get_config(self) -> dict:
        __config = super(HeadBlock, self).get_config()
        __config.update(self._config)
        return __config

    @classmethod
    def from_config(cls, config) -> tf.keras.layers.Layer:
        return cls(**config)
