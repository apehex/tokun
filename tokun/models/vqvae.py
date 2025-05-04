"""Piece together the actual VAE CNN model for tokun."""

import functools

import keras
import tensorflow as tf

import mlable.layers.embedding
import mlable.layers.shaping

# ENCODER #####################################################################

@keras.saving.register_keras_serializable(package='models')
class Encoder(tf.keras.models.Model):
    def __init__(
        self,
        embed_dim: int,
        input_dim: int=256,
        trainable: bool=True,
        **kwargs
    ) -> None:
        # init
        super(Encoder, self).__init__(**kwargs)
        # config
        self._config = {
            'input_dim': input_dim,
            'embed_dim': embed_dim,
            'trainable': trainable,}
        # layers
        self._layers = []

    def build(self, input_shape: tuple) -> None:
        __shape = tuple(input_shape)
        # weights
        __q, __r = tf.linalg.qr(tf.random.normal((self._config['input_dim'], self._config['embed_dim'])))
        __e = __q * tf.sign(tf.linalg.diag_part(tf.expand_dims(__r, axis=0)))
        # init
        self._layers = [
            # (B, ..., T) => (B, ..., T, E)
            tf.keras.layers.Embedding(
                input_dim=self._config['input_dim'],
                output_dim=self._config['embed_dim'],
                embeddings_initializer=tf.keras.initializers.Constant(__e),
                name='encoder-0-embed'),
            # (B, ..., T, E) => (B, ..., T * E)
            mlable.layers.shaping.Merge(
                axis=-1,
                right=False),]
        # build
        for __l in self._layers:
            __l.build(__shape)
            __shape = __l.compute_output_shape(__shape)
        # freeze
        for __l in self._layers:
            __l.trainable = self._config['trainable']
        # register
        self.built = True

    def compute_embedding_kl_loss(self, beta: float=1.0) -> tf.Tensor:
        __e = self._layers[0].embeddings
        # mean (E,)
        __m = tf.reduce_mean(__e, axis=0)
        # variance (E,)
        __v = tf.math.reduce_variance(__e, axis=0)
        # KL divergence ()
        return beta * 0.5 * tf.reduce_sum(__v + tf.square(__m) - 1. - tf.math.log(__v + 1e-8))

    def compute_output_shape(self, input_shape: tuple) -> tuple:
        return functools.reduce(lambda __s, __l: __l.compute_output_shape(__s), self._layers, input_shape)

    def call(self, inputs: tf.Tensor, training: bool=False, **kwargs) -> tf.Tensor:
        # KL divergence of the embeddings to N(0, I)
        if training:
            self.add_loss(tf.cast(self.compute_embedding_kl_loss(beta=1.0), dtype=tf.float32))
        # composite embeddings, token by token (B, ..., T) => (B, ..., T * E)
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
        token_dim: int,
        binary_dim: int=8, # return byte predictions
        **kwargs
    ) -> None:
        # init
        super(Decoder, self).__init__(**kwargs)
        # config
        self._config = {
            'token_dim': token_dim,
            'binary_dim': binary_dim,}
        # layers
        self._layers = []

    def build(self, input_shape: tuple) -> tuple:
        __shape = tuple(input_shape)
        # init
        self._layers = [
            # (B, ..., L) => (B, ..., T * O)
            tf.keras.layers.Dense(
                units=self._config['token_dim'] * self._config['binary_dim'],
                use_bias=True,
                activation=None,
                kernel_initializer='zeros',
                bias_initializer='zeros',
                name='projection')]
        # build
        for __l in self._layers:
            __l.build(__shape)
            __shape = __l.compute_output_shape(__shape)
        # register
        self.built = True

    def compute_output_shape(self, input_shape: tuple) -> tuple:
        return functools.reduce(lambda __s, __l: __l.compute_output_shape(__s), self._layers, input_shape)

    def call(self, inputs: tf.Tensor, logits: bool=True, training: bool=False, **kwargs) -> tf.Tensor:
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
class QuantizedAutoEncoder(tf.keras.models.Model):
    def __init__(
        self,
        token_dim: int,
        embed_dim: int,
        input_dim: int=256, # byte inputs
        binary_dim: int=8, # byte predictions
        trainable: bool=True,
        **kwargs
    ) -> None:
        # init
        super(QuantizedAutoEncoder, self).__init__(**kwargs)
        # config
        self._config = {
            'token_dim': token_dim,
            'input_dim': input_dim,
            'embed_dim': embed_dim,
            'binary_dim': binary_dim,
            'trainable': trainable,}
        # layers
        self._encoder = None
        self._decoder = None

    def build(self, input_shape: tuple) -> None:
        __shape = tuple(input_shape)
        # init
        self._encoder = Encoder(
            input_dim=self._config['input_dim'],
            embed_dim=self._config['embed_dim'],
            trainable=self._config['trainable'])
        self._decoder = Decoder(
            token_dim=self._config['token_dim'],
            binary_dim=self._config['binary_dim'],)
        # build
        self._encoder.build(__shape)
        __shape = self._encoder.compute_output_shape(__shape)
        self._decoder.build(__shape)
        __shape = self._decoder.compute_output_shape(__shape)
        # register
        self.built = True

    def compute_latent_shape(self, input_shape: tuple) -> tuple:
        return self._encoder.compute_output_shape(input_shape)

    def compute_output_shape(self, input_shape: tuple) -> tuple:
        return self._decoder.compute_output_shape(self._encoder.compute_output_shape(input_shape))

    def encode(self, inputs: tf.Tensor, training: bool=False, **kwargs) -> tf.Tensor:
        return self._encoder(inputs, training=training, **kwargs)

    def decode(self, inputs: tf.Tensor, logits: bool=True, training: bool=False, **kwargs) -> tf.Tensor:
        return self._decoder(inputs, logits=logits, training=training, **kwargs)

    def call(self, inputs: tf.Tensor, logits: bool=True, training: bool=False, **kwargs) -> tf.Tensor:
        # composite embeddings (B, ..., T) => (B, ..., T * E)
        __outputs = self._encoder(inputs, training=training, **kwargs)
        # binary encoding of each token (B, ..., T * E) => (B, ..., T * U)
        return self._decoder(__outputs, logits=logits, training=training, **kwargs)

    def get_config(self) -> dict:
        __config = super(QuantizedAutoEncoder, self).get_config()
        __config.update(self._config)
        return __config

    @classmethod
    def from_config(cls, config) -> tf.keras.layers.Layer:
        return cls(**config)
