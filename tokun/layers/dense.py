"""Encoding and decoding blocks of tokun."""

import functools

import keras
import tensorflow as tf

import mlable.layers.shaping
import mlable.layers.transformer

# ENCODING BLOCKS #############################################################

@keras.saving.register_keras_serializable(package='blocks')
class TokenizeBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        sequence_axis: int=1,
        feature_axis: int=-1,
        token_dim: int=4,
        latent_dim: int=4,
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
            'latent_dim': latent_dim,
            'activation': activation,
            'epsilon': epsilon,}
        # layers
        self._layers = []

    def build(self, input_shape: tuple) -> None:
        __shape = tuple(input_shape)
        # init
        self._layers = [
            # normalize each token unit independently
            tf.keras.layers.LayerNormalization(
                axis=self._config['feature_axis'],
                epsilon=self._config['epsilon'],
                center=True,
                scale=True,
                name='normalization'),
            # (B, S * G, E) => (B, S, G * E)
            mlable.layers.shaping.Divide(
                axis=self._config['sequence_axis'],
                factor=self._config['token_dim'],
                insert=False,
                right=True,
                name='reshaping'),
            # (B, S, G * E) => (B, S, L), typically L = E
            tf.keras.layers.Dense(
                units=self._config['latent_dim'],
                activation=self._config['activation'],
                use_bias=True,
                kernel_initializer='glorot_uniform',
                bias_initializer='zeros',
                name='compression'),]
        # build
        for __l in self._layers:
            __l.build(__shape)
            __shape = __l.compute_output_shape(__shape)
        # register
        self.built = True

    def compute_output_shape(self, input_shape: tuple) -> tuple:
        return functools.reduce(lambda __s, __l: __l.compute_output_shape(__s), self._layers, input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return functools.reduce(lambda __t, __l: __l(__t), self._layers, inputs)

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
        sequence_axis: int=1,
        feature_axis: int=-1,
        token_dim: int=4,
        latent_dim: int=4,
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
            'latent_dim': latent_dim,
            'activation': activation,
            'epsilon': epsilon,}
        # layers
        self._layers = []

    def build(self, input_shape: tuple) -> None:
        __shape = tuple(input_shape)
        # init
        self._layers = [
            # (B, S, L) => (B, S, G * E), typically L = E
            tf.keras.layers.Dense(
                units=self._config['token_dim'] * self._config['latent_dim'],
                activation=self._config['activation'],
                use_bias=True,
                kernel_initializer='glorot_uniform',
                bias_initializer='zeros',
                name='decompression'),
            # (B, S, G * E) => (B, S * G, E)
            mlable.layers.shaping.Divide(
                axis=self._config['feature_axis'],
                factor=self._config['token_dim'],
                insert=False,
                right=False,
                name='reshaping'),
            # normalize each token unit independently
            tf.keras.layers.LayerNormalization(
                axis=self._config['feature_axis'],
                epsilon=self._config['epsilon'],
                center=True,
                scale=True,
                name='normalization'),]
        # build
        for __l in self._layers:
            __l.build(__shape)
            __shape = __l.compute_output_shape(__shape)
        # register
        self.built = True

    def compute_output_shape(self, input_shape: tuple) -> tuple:
        return functools.reduce(lambda __s, __l: __l.compute_output_shape(__s), self._layers, input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return functools.reduce(lambda __t, __l: __l(__t), self._layers, inputs)

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
        output_dim: int=8,
        **kwargs
    ) -> None:
        super(HeadBlock, self).__init__(**kwargs)
        # config
        self._config = {'output_dim': output_dim,}
        # layers
        self._dense = None # (..., G, E) => (..., G, U), typically U = E

    def build(self, input_shape: tuple) -> None:
        __shape = tuple(input_shape)
        # init
        self._dense = tf.keras.layers.Dense(
            units=self._config['output_dim'],
            use_bias=True,
            activation='linear',
            kernel_initializer='zeros',
            bias_initializer='zeros',
            name='projection')
        # build
        self._dense.build(__shape)
        # register
        self.built = True

    def compute_output_shape(self, input_shape: tuple) -> tuple:
        return tuple(input_shape[:-1]) + (self._config['output_dim'],)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return self._dense(inputs)

    def get_config(self) -> dict:
        __config = super(HeadBlock, self).get_config()
        __config.update(self._config)
        return __config

    @classmethod
    def from_config(cls, config) -> tf.keras.layers.Layer:
        return cls(**config)
