"""Piece together the actual VAE CNN model for tokun."""

import functools

import keras
import tensorflow as tf

import tokun.layers.dense

# ENCODER #####################################################################

@keras.saving.register_keras_serializable(package='models')
class Encoder(tf.keras.models.Model):
    def __init__(
        self,
        token_dim: list,
        latent_dim: list,
        embed_dim: int,
        input_dim: int,
        sequence_axis: int=1,
        feature_axis: int=-1,
        activation: str='gelu',
        **kwargs
    ) -> None:
        # init
        super(Encoder, self).__init__(**kwargs)
        # config
        self._config = {
            'token_dim': [token_dim] if isinstance(token_dim, int) else list(token_dim),
            'latent_dim': [latent_dim] if isinstance(latent_dim, int) else list(latent_dim),
            'input_dim': input_dim,
            'embed_dim': embed_dim,
            'sequence_axis': sequence_axis,
            'feature_axis': feature_axis,
            'activation': activation,}
        # layers
        self._layers = []

    def build(self, input_shape: tuple) -> None:
        __shape = tuple(input_shape)
        # init
        self._layers = [
            # (B, G ^ D) => (B, G ^ D, E)
            tf.keras.layers.Embedding(
                input_dim=self._config['input_dim'],
                output_dim=self._config['embed_dim'],
                embeddings_initializer='glorot_uniform',
                name='encoder-0-embed'),] + [
            # (B, G ^ i, E) => (B, G ^ (i-1), E)
            tokun.layers.dense.TokenizeBlock(
                sequence_axis=self._config['sequence_axis'],
                feature_axis=self._config['feature_axis'],
                token_dim=__t,
                latent_dim=__e,
                activation=self._config['activation'],
                name='encoder-1-tokenize-{}_{}-{}'.format(__i, __t, __e))
            for __i, (__t, __e) in enumerate(zip(self._config['token_dim'], self._config['latent_dim']))]
        # build
        for __l in self._layers:
            __l.build(__shape)
            __shape = __l.compute_output_shape(__shape)
        # register
        self.built = True

    def compute_output_shape(self, input_shape: tuple) -> tuple:
        return functools.reduce(lambda __s, __l: __l.compute_output_shape(__s), self._layers, input_shape)

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        return functools.reduce(lambda __x, __l: __l(__x, **kwargs), self._layers, inputs)

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
        latent_dim: list,
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
            'token_dim': [token_dim] if isinstance(token_dim, int) else list(token_dim),
            'latent_dim': [latent_dim] if isinstance(latent_dim, int) else list(latent_dim),
            'output_dim': output_dim,
            'sequence_axis': sequence_axis,
            'feature_axis': feature_axis,
            'activation': activation,}
        # layers
        self._layers = []

    def build(self, input_shape: tuple) -> tuple:
        __shape = tuple(input_shape)
        # init
        self._layers = [
            # (B, G ^ i, E) => (B, G ^ (i+1), E)
            tokun.layers.dense.DetokenizeBlock(
                sequence_axis=self._config['sequence_axis'],
                feature_axis=self._config['feature_axis'],
                token_dim=__t,
                latent_dim=__e,
                activation=self._config['activation'],
                name='decoder-1-detokenize-{}_{}-{}'.format(__i, __t, __e))
            for __i, (__t, __e) in enumerate(zip(self._config['token_dim'], self._config['latent_dim']))] + [
            # (B, G ^ D, E) => (B, G ^ D, U)
            tokun.layers.dense.HeadBlock(
                output_dim=self._config['output_dim'],
                name='project-head')]
        # build
        for __l in self._layers:
            __l.build(__shape)
            __shape = __l.compute_output_shape(__shape)
        # register
        self.built = True

    def compute_output_shape(self, input_shape: tuple) -> tuple:
        return functools.reduce(lambda __s, __l: __l.compute_output_shape(__s), self._layers, input_shape)

    def call(self, inputs: tf.Tensor, logits: bool=True, **kwargs) -> tf.Tensor:
        __outputs = functools.reduce(lambda __x, __l: __l(__x, **kwargs), self._layers, inputs)
        # raw logits or probabilities
        return __outputs if logits else tf.sigmoid(__outputs)

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
        latent_dim: list,
        input_dim: int,
        embed_dim: int,
        output_dim: int,
        sequence_axis: int=1,
        feature_axis: int=-1,
        activation: str='gelu',
        **kwargs
    ) -> None:
        # init
        super(AutoEncoder, self).__init__(**kwargs)
        # config
        self._config = {
            'token_dim': [token_dim] if isinstance(token_dim, int) else list(token_dim),
            'latent_dim': [latent_dim] if isinstance(latent_dim, int) else list(latent_dim),
            'input_dim': input_dim,
            'embed_dim': embed_dim,
            'output_dim': output_dim,
            'sequence_axis': sequence_axis,
            'feature_axis': feature_axis,
            'activation': activation,}
        # layers
        self._encoder = None
        self._decoder = None

    def build(self, input_shape: tuple) -> None:
        __shape = tuple(input_shape)
        # init
        self._encoder = Encoder(
            token_dim=self._config['token_dim'],
            latent_dim=self._config['latent_dim'],
            input_dim=self._config['input_dim'],
            embed_dim=self._config['embed_dim'],
            sequence_axis=self._config['sequence_axis'],
            feature_axis=self._config['feature_axis'],
            activation=self._config['activation'])
        self._decoder = Decoder(
            token_dim=self._config['token_dim'][::-1],
            output_dim=self._config['output_dim'],
            latent_dim=self._config['latent_dim'],
            sequence_axis=self._config['sequence_axis'],
            feature_axis=self._config['feature_axis'],
            activation=self._config['activation'])
        # build
        self._encoder.build(__shape)
        __shape = self._encoder.compute_output_shape(__shape)
        self._decoder.build(__shape)
        __shape = self._decoder.compute_output_shape(__shape)
        # register
        self.built = True

    def compute_output_shape(self, input_shape: tuple) -> tuple:
        return tuple(input_shape) + (self._config['output_dim'],)

    def encode(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        return self._encoder(inputs, **kwargs)

    def decode(self, inputs: tf.Tensor, logits: bool=True, **kwargs) -> tf.Tensor:
        return self._decoder(inputs, logits=logits, **kwargs)

    def call(self, inputs: tf.Tensor, logits: bool=True, **kwargs) -> tf.Tensor:
        return self._decoder(self._encoder(inputs, **kwargs), logits=logits, **kwargs)

    def get_config(self) -> dict:
        __config = super(AutoEncoder, self).get_config()
        __config.update(self._config)
        return __config

    @classmethod
    def from_config(cls, config) -> tf.keras.layers.Layer:
        return cls(**config)
