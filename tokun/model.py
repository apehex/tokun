"""Piece together the actual VAE CNN model for tokun."""

import keras
import tensorflow as tf

import tokun.layers

# ENCODER #####################################################################

@keras.saving.register_keras_serializable(package='models')
class Encoder(tf.keras.models.Model):
    def __init__(
        self,
        token_dim: list,
        encoding_dim: int,
        embedding_dim: int,
        hidden_dim: int,
        latent_dim: int,
        activation: str='gelu',
        sequence_axis: int=0,
        feature_axis: int=-1,
        **kwargs
    ) -> None:
        # init
        super(Encoder, self).__init__(**kwargs)
        # config
        self._config = {
            'token_dim': token_dim,
            'encoding_dim': encoding_dim,
            'embedding_dim': embedding_dim,
            'hidden_dim': hidden_dim,
            'latent_dim': latent_dim,
            'activation': activation,
            'sequence_axis': sequence_axis,
            'feature_axis': feature_axis,}
        # successive dimensions of the merging units
        __token_dim = [token_dim] if isinstance(token_dim, int) else token_dim
        # layers
        __layers = [
            # (B * G ^ D, U) => (B * G ^ D, E)
            tf.keras.layers.Embedding(
                input_dim=encoding_dim,
                output_dim=embedding_dim,
                embeddings_initializer='glorot_uniform',
                name='embed-1'),] + [
            # (B * G ^ i, E) => (B * G ^ (i-1), E)
            tokun.layers.TokenizeBlock(
                sequence_axis=sequence_axis,
                feature_axis=feature_axis,
                token_dim=__g,
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
                latent_dim=latent_dim,
                activation=activation,
                name='tokenize-{}_{}'.format(__g, __i))
            for __i, __g in enumerate(__token_dim)]
        # model
        self._encoder = tf.keras.Sequential(__layers)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        return self._encoder(x)

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
        encoding_dim: int,
        embedding_dim: int,
        hidden_dim: int,
        latent_dim: int,
        activation: str='gelu',
        sequence_axis: int=0,
        feature_axis: int=-1,
        **kwargs
    ) -> None:
        # init
        super(Decoder, self).__init__(**kwargs)
        # config
        self._config = {
            'token_dim': token_dim,
            'encoding_dim': encoding_dim,
            'embedding_dim': embedding_dim,
            'hidden_dim': hidden_dim,
            'latent_dim': latent_dim,
            'activation': activation,
            'sequence_axis': sequence_axis,
            'feature_axis': feature_axis,}
        # successive dimensions of the dividing units
        __token_dim = [token_dim] if isinstance(token_dim, int) else token_dim
        # layers
        __layers = [
            # (B * G ^ i, E) => (B * G ^ (i+1), E)
            tokun.layers.DetokenizeBlock(
                sequence_axis=sequence_axis,
                feature_axis=feature_axis,
                token_dim=__g,
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
                activation=activation,
                name='detokenize-{}_{}'.format(__g, __i))
            for __i, __g in enumerate(__token_dim)] + [
            # (B * G ^ D, E) => (B * G ^ D, U)
            tokun.layers.HeadBlock(feature_axis=feature_axis, encoding_dim=encoding_dim, name='project-head')]
        # model
        self._decoder = tf.keras.Sequential(__layers) 

    def call(self, x: tf.Tensor) -> tf.Tensor:
        return self._decoder(x)

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
        encoding_dim: int,
        embedding_dim: int,
        hidden_dim: int,
        latent_dim: int,
        activation: str='gelu',
        sequence_axis: int=0,
        feature_axis: int=-1,
        **kwargs
    ) -> None:
        # init
        super(AutoEncoder, self).__init__(**kwargs)
        # layers
        self._encoder = Encoder(token_dim=token_dim, encoding_dim=encoding_dim, embedding_dim=embedding_dim, hidden_dim=hidden_dim, latent_dim=latent_dim, activation=activation, sequence_axis=sequence_axis, feature_axis=feature_axis)
        self._decoder = Decoder(token_dim=token_dim[::-1], encoding_dim=encoding_dim, embedding_dim=embedding_dim, hidden_dim=hidden_dim, latent_dim=latent_dim, activation=activation, sequence_axis=sequence_axis, feature_axis=feature_axis)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        return self._decoder(self._encoder(x))

    def get_config(self) -> dict:
        __config = super(AutoEncoder, self).get_config()
        __config.update(self._encoder.get_config())
        return __config

    @classmethod
    def from_config(cls, config) -> tf.keras.layers.Layer:
        return cls(**config)
