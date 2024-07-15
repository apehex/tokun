"""Encoding and decoding blocks of tokun."""

import keras
import tensorflow as tf

import mlable.layers.reshaping
import mlable.layers.transformer

# ENCODING BLOCKS #############################################################

@keras.saving.register_keras_serializable(package='blocks')
class TokenizeBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        sequence_axis: int=0,
        feature_axis: int=-1,
        token_dim: int=4,
        embedding_dim: int=256,
        activation: str='gelu',
        epsilon: float=1e-6,
        **kwargs
    ) -> None:
        super(TokenizeBlock, self).__init__(**kwargs)
        # config
        self._config = {
            'sequence_axis': sequence_axis,
            'feature_axis': feature_axis,
            'token_dim': token_dim,
            'embedding_dim': embedding_dim,
            'activation': activation,
            'epsilon': epsilon,}
        # layers
        self._normalize = tf.keras.layers.LayerNormalization(axis=feature_axis, epsilon=epsilon, center=True, scale=True, name='normalization') # normalize each token unit independently
        self._divide = mlable.layers.reshaping.Divide(input_axis=sequence_axis, output_axis=feature_axis, factor=token_dim, insert=False, name='reshaping') # (B, S * G, E) => (B, S, G * E)
        self._dense = tf.keras.layers.Dense(units=embedding_dim, activation=activation, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', name='compression') # (B, S, G * E) => (B, S, L), typically L = E

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return self._dense(self._divide(self._normalize(inputs)))

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
        sequence_axis: int=0,
        feature_axis: int=-1,
        token_dim: int=4,
        embedding_dim: int=256,
        activation: str='gelu',
        epsilon: float=1e-6,
        **kwargs
    ) -> None:
        super(DetokenizeBlock, self).__init__(**kwargs)
        # config
        self._config = {
            'sequence_axis': sequence_axis,
            'feature_axis': feature_axis,
            'token_dim': token_dim,
            'embedding_dim': embedding_dim,
            'activation': activation,
            'epsilon': epsilon,}
        # layers
        self._dense = tf.keras.layers.Dense(units=token_dim * embedding_dim, activation=activation, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', name='decompression') # (B, S, L) => (B, S, G * E), typically L = E
        self._divide = mlable.layers.reshaping.Divide(input_axis=feature_axis, output_axis=sequence_axis, insert=False, factor=token_dim, name='reshaping') # (B, S, G * E) => (B, S * G, E)
        self._normalize = tf.keras.layers.LayerNormalization(axis=feature_axis, epsilon=epsilon, center=True, scale=True, name='normalization') # normalize each token unit independently

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return self._normalize(self._divide(self._dense(inputs)))

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
        encoding_dim: int=256,
        activation: str='softmax',
        **kwargs
    ) -> None:
        super(HeadBlock, self).__init__(**kwargs)
        # config
        self._config = {'encoding_dim': encoding_dim, 'activation': activation}
        # layers
        self._dense = tf.keras.layers.Dense(units=encoding_dim, activation=activation, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', name='projection') # (..., G, E) => (..., G, U), typically U = E

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return self._dense(inputs)

    def get_config(self) -> dict:
        __config = super(HeadBlock, self).get_config()
        __config.update(self._config)
        return __config

    @classmethod
    def from_config(cls, config) -> tf.keras.layers.Layer:
        return cls(**config)
