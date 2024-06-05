"""Piece together the actual VAE CNN model for tokun."""

import keras
import tensorflow as tf

import tokun.layers

# ENCODER #####################################################################

@keras.saving.register_keras_serializable(package='models')
class Encoder(tf.keras.models.Model):
    def __init__(self, token_dim: list, encoding_dim: int, embedding_dim: int, latent_dim: int, batch_dim: int=None, attention: bool=False, normalization: bool=False, activation: str='silu', **kwargs) -> None:
        # init
        super(Encoder, self).__init__(**kwargs)
        # config
        self._config = {
            'token_dim': token_dim,
            'encoding_dim': encoding_dim,
            'embedding_dim': embedding_dim,
            'latent_dim': latent_dim,
            'batch_dim': batch_dim,
            'attention': attention,
            'normalization': normalization,
            'activation': activation}
        # successive dimensions of the merging units
        __token_dim = [token_dim] if isinstance(token_dim, int) else token_dim
        # layers
        self._encoder = tf.keras.Sequential([
            tf.keras.Input(shape=(encoding_dim,), batch_size=batch_dim, name='input'), # (B * G ^ D, U)
            tf.keras.layers.Dense(units=embedding_dim, activation=None, use_bias=False, kernel_initializer='glorot_uniform', bias_initializer=None, name='embed-1'),] # (B * G ^ D, U) => (B * G ^ D, E)
            + [tokun.layers.TokenizeBlock(feature_axis=-1, token_dim=__g, latent_dim=latent_dim, attention=attention, normalization=normalization, activation=activation, name='tokenize-{}_{}'.format(__g, __i)) for __i, __g in enumerate(__token_dim)]) # (B * G ^ i, E) => (B * G ^ (i-1), E)

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
    def __init__(self, token_dim: list, encoding_dim: int, embedding_dim: int, latent_dim: int, batch_dim: int=None, attention: bool=False, normalization: bool=False, activation: str='silu', **kwargs) -> None:
        # init
        super(Decoder, self).__init__(**kwargs)
        # config
        self._config = {
            'token_dim': token_dim,
            'encoding_dim': encoding_dim,
            'embedding_dim': embedding_dim,
            'latent_dim': latent_dim,
            'batch_dim': batch_dim,
            'attention': attention,
            'normalization': normalization,
            'activation': activation}
        # successive dimensions of the dividing units
        __token_dim = [token_dim] if isinstance(token_dim, int) else token_dim
        # layers
        self._decoder = tf.keras.Sequential(
            [tf.keras.Input(shape=(latent_dim,), batch_size=batch_dim, name='input')] # (B, E)
            + [tokun.layers.DetokenizeBlock(feature_axis=-1, token_dim=__g, embedding_dim=embedding_dim, attention=attention, normalization=normalization, activation=activation, name='detokenize-{}_{}'.format(__g, __i)) for __i, __g in enumerate(__token_dim)] # (B * G ^ i, E) => (B * G ^ (i+1), E)
            + [tokun.layers.HeadBlock(feature_axis=-1, encoding_dim=encoding_dim, name='project-head')]) # (B * G ^ D, E) => (B * G ^ D, U)

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
    def __init__(self, token_dim: list, encoding_dim: int, embedding_dim: int, latent_dim: int, batch_dim: int=None, attention: bool=False, normalization: bool=False, activation: str='silu', **kwargs) -> None:
        # init
        super(AutoEncoder, self).__init__(**kwargs)
        # layers
        self._encoder = Encoder(token_dim=token_dim, encoding_dim=encoding_dim, embedding_dim=embedding_dim, latent_dim=latent_dim, batch_dim=batch_dim, attention=attention, normalization=normalization, activation=activation)
        self._decoder = Decoder(token_dim=token_dim, encoding_dim=encoding_dim, embedding_dim=embedding_dim, latent_dim=latent_dim, batch_dim=batch_dim, attention=attention, normalization=normalization, activation=activation)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        return self._decoder(self._encoder(x))

    def get_config(self) -> dict:
        __config = super(AutoEncoder, self).get_config()
        __config.update(self._encoder.get_config())
        return __config

    @classmethod
    def from_config(cls, config) -> tf.keras.layers.Layer:
        return cls(**config)
