"""Piece together the actual VAE CNN model for tokun."""

import keras
import tensorflow as tf

import tokun.layers

# ENCODER #####################################################################

@keras.saving.register_keras_serializable(package='models')
class Encoder(tf.keras.models.Model):
    def __init__(self, token_dim: list, encoding_dim: int, embedding_dim: int, latent_dim: int, batch_dim: int=None, attention: bool=False, normalization: bool=False, **kwargs) -> None:
        super(Encoder, self).__init__(**kwargs)
        self._encoder = tf.keras.Sequential([
            tf.keras.Input(shape=(encoding_dim,), batch_size=batch_dim, name='input'), # (B * G ^ D, U)
            tf.keras.layers.Dense(units=embedding_dim, activation=None, use_bias=False, kernel_initializer='glorot_uniform', bias_initializer=None, name='embed-1'),] # (B * G ^ D, U) => (B * G ^ D, E)
            + [tokun.layers.TokenizeBlock(left_axis=-2, right_axis=-1, token_dim=__g, latent_dim=latent_dim, attention=attention, normalization=normalization, name='tokenize-{}_{}'.format(__g, __i)) for __i, __g in enumerate(token_dim)]) # (B * G ^ i, E) => (B * G ^ (i-1), E)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        return self._encoder(x)

    def get_config(self) -> dict:
        __parent_config = super(Encoder, self).get_config()
        __input_shape = list(self._encoder.inputs[0].shape)
        __embedding_config = self._encoder.layers[0].get_config()
        __tokenizer_config = self._encoder.layers[1].get_config()
        __token_dim = [__b.get_config().get('token_dim', 4) for __b in self._encoder.layers[1:]]
        __child_config = {
            'batch_dim': __input_shape[0],
            'encoding_dim': __input_shape[-1],
            'embedding_dim': __embedding_config['units'],
            'token_dim': __token_dim,
            'latent_dim': __tokenizer_config['latent_dim'],
            'attention': __tokenizer_config['attention'],
            'normalization': __tokenizer_config['normalization'],}
        return {**__parent_config, **__child_config}

    @classmethod
    def from_config(cls, config) -> tf.keras.layers.Layer:
        return cls(**config)

# DECODER #####################################################################

@keras.saving.register_keras_serializable(package='models')
class Decoder(tf.keras.models.Model):
    def __init__(self, token_dim: list, encoding_dim: int, embedding_dim: int, latent_dim: int, batch_dim: int=None, attention: bool=False, normalization: bool=False, **kwargs) -> None:
        super(Decoder, self).__init__(**kwargs)
        self._decoder = tf.keras.Sequential(
            [tf.keras.Input(shape=(latent_dim,), batch_size=batch_dim, name='input')] # (B, E)
            + [tokun.layers.DetokenizeBlock(token_dim=__g, embedding_dim=embedding_dim, attention=attention, normalization=normalization, name='detokenize-{}_{}'.format(__g, __i)) for __i, __g in enumerate(token_dim)] # (B * G ^ i, E) => (B * G ^ (i+1), E)
            + [tokun.layers.HeadBlock(encoding_dim=encoding_dim, name='project-head')]) # (B * G ^ D, E) => (B * G ^ D, U)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        return self._decoder(x)

    def get_config(self) -> dict:
        __parent_config = super(Decoder, self).get_config()
        __input_shape = list(self._decoder.inputs[0].shape)
        __detokenizer_config = self._decoder.layers[0].get_config()
        __head_config = self._decoder.layers[-1].get_config()
        __token_dim = [__b.get_config().get('token_dim', 4) for __b in self._encoder.layers[:-1]]
        __child_config = {
            'batch_dim': __input_shape[0],
            'latent_dim': __input_shape[-1],
            'encoding_dim': __head_config['encoding_dim'],
            'token_dim': __detokenizer_config['token_dim'],
            'embedding_dim': __detokenizer_config['embedding_dim'],
            'attention': __detokenizer_config['attention'],
            'normalization': __detokenizer_config['normalization'],}
        return {**__parent_config, **__child_config}

    @classmethod
    def from_config(cls, config) -> tf.keras.layers.Layer:
        return cls(**config)

# VAE #########################################################################

@keras.saving.register_keras_serializable(package='models')
class AutoEncoder(tf.keras.models.Model):
    def __init__(self, token_dim: list, encoding_dim: int, embedding_dim: int, latent_dim: int, batch_dim: int=None, attention: bool=False, normalization: bool=False, **kwargs) -> None:
        super(AutoEncoder, self).__init__(**kwargs)
        self._encoder = Encoder(token_dim=token_dim, encoding_dim=encoding_dim, embedding_dim=embedding_dim, latent_dim=latent_dim, batch_dim=batch_dim, attention=attention, normalization=normalization)
        self._decoder = Decoder(token_dim=token_dim, encoding_dim=encoding_dim, embedding_dim=embedding_dim, latent_dim=latent_dim, batch_dim=batch_dim, attention=attention, normalization=normalization)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        return self._decoder(self._encoder(x))

    def get_config(self) -> dict:
        __parent_config = super(AutoEncoder, self).get_config()
        __encoder_config = self._encoder.get_config()
        return {**__encoder_config, **__parent_config}

    @classmethod
    def from_config(cls, config) -> tf.keras.layers.Layer:
        return cls(**config)
