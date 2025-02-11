"""Piece together the actual VAE CNN model for tokun."""

import functools

import keras
import tensorflow as tf

import mlable.layers.shaping
import tokun.layers.dense

# ENCODER #####################################################################

@keras.saving.register_keras_serializable(package='models')
class Encoder(tf.keras.models.Model):
    def __init__(
        self,
        token_dim: list,
        input_dim: int,
        latent_dim: int,
        sequence_axis: int=1,
        feature_axis: int=-1,
        activation: str='gelu',
        **kwargs
    ) -> None:
        # init
        super(Encoder, self).__init__(**kwargs)
        # config
        self._config = {
            'token_dim': token_dim,
            'input_dim': input_dim,
            'latent_dim': latent_dim,
            'sequence_axis': sequence_axis,
            'feature_axis': feature_axis,
            'activation': activation,}
        # successive dimensions of the merging units
        __token_dim = [token_dim] if isinstance(token_dim, int) else token_dim
        # layers
        self._layers = [
            # (B * G ^ D, U) => (B * G ^ D, E)
            tf.keras.layers.Embedding(
                input_dim=input_dim,
                output_dim=latent_dim,
                embeddings_initializer='glorot_uniform',
                name='embed-1'),] + [
            # (B * G ^ i, E) => (B * G ^ (i-1), E)
            tokun.layers.dense.TokenizeBlock(
                sequence_axis=sequence_axis,
                feature_axis=feature_axis,
                token_dim=__g,
                latent_dim=latent_dim,
                activation=activation,
                name='tokenize-{}_{}'.format(__g, __i))
            for __i, __g in enumerate(__token_dim)]

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        return functools.reduce(lambda __x, __l: __l(__x), self._layers, inputs)

    def get_config(self) -> dict:
        __config = super(Encoder, self).get_config()
        __config.update(self._config)
        return __config

    @classmethod
    def from_config(cls, config) -> tf.keras.layers.Layer:
        return cls(**config)

# DECODER #####################################################################

@keras.saving.register_keras_serializable(package='models')
class Decoder(tf.keras.models.Model):
    def __init__(
        self,
        token_dim: list,
        latent_dim: int,
        output_dim: int,
        sequence_axis: int=1,
        feature_axis: int=-1,
        activation: str='gelu',
        **kwargs
    ) -> None:
        # init
        super(Decoder, self).__init__(**kwargs)
        # config
        self._config = {
            'token_dim': token_dim,
            'latent_dim': latent_dim,
            'output_dim': output_dim,
            'sequence_axis': sequence_axis,
            'feature_axis': feature_axis,
            'activation': activation,}
        # successive dimensions of the dividing units
        __token_dim = [token_dim] if isinstance(token_dim, int) else token_dim
        # layers
        self._layers = [
            # (B * G ^ i, E) => (B * G ^ (i+1), E)
            tokun.layers.dense.DetokenizeBlock(
                sequence_axis=sequence_axis,
                feature_axis=feature_axis,
                token_dim=__g,
                latent_dim=latent_dim,
                activation=activation,
                name='detokenize-{}_{}'.format(__g, __i))
            for __i, __g in enumerate(__token_dim)] + [
            # (B * G ^ D, E) => (B * G ^ D, U)
            tokun.layers.dense.HeadBlock(output_dim=output_dim, name='project-head')]

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        __outputs = functools.reduce(lambda __x, __l: __l(__x), self._layers, inputs)
        # raw logits or probabilities
        return __outputs if kwargs.get('raw', True) else tf.sigmoid(__outputs)

    def get_config(self) -> dict:
        __config = super(Decoder, self).get_config()
        __config.update(self._config)
        return __config

    @classmethod
    def from_config(cls, config) -> tf.keras.layers.Layer:
        return cls(**config)

# VAE #########################################################################

@keras.saving.register_keras_serializable(package='models')
class AutoEncoder(tf.keras.models.Model):
    def __init__(
        self,
        token_dim: list,
        input_dim: int,
        output_dim: int,
        latent_dim: int,
        sequence_axis: int=1,
        feature_axis: int=-1,
        activation: str='gelu',
        **kwargs
    ) -> None:
        # init
        super(AutoEncoder, self).__init__(**kwargs)
        # config
        self._config = {
            'token_dim': token_dim,
            'input_dim': input_dim,
            'latent_dim': latent_dim,
            'output_dim': output_dim,
            'sequence_axis': sequence_axis,
            'feature_axis': feature_axis,
            'activation': activation,}
        # layers
        self._encoder = Encoder(token_dim=token_dim, input_dim=input_dim, latent_dim=latent_dim, sequence_axis=sequence_axis, feature_axis=feature_axis, activation=activation)
        self._decoder = Decoder(token_dim=token_dim[::-1], output_dim=output_dim, latent_dim=latent_dim, sequence_axis=sequence_axis, feature_axis=feature_axis, activation=activation)

    def encode(self, inputs: tf.Tensor) -> tf.Tensor:
        return self._encoder(inputs)

    def decode(self, inputs: tf.Tensor) -> tf.Tensor:
        return self._decoder(inputs)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return self._decoder(self._encoder(inputs))

    def get_config(self) -> dict:
        __config = super(AutoEncoder, self).get_config()
        __config.update(self._config)
        return __config

    @classmethod
    def from_config(cls, config) -> tf.keras.layers.Layer:
        return cls(**config)
