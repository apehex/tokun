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
            'encoding_dim': encoding_dim,
            'embedding_dim': embedding_dim,
            'sequence_axis': sequence_axis,
            'feature_axis': feature_axis,
            'activation': activation,}
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
        sequence_axis: int=1,
        feature_axis: int=-1,
        activation: str='gelu',
        output: str='categorical',
        **kwargs
    ) -> None:
        # init
        super(Decoder, self).__init__(**kwargs)
        # config
        self._config = {
            'token_dim': token_dim,
            'encoding_dim': encoding_dim,
            'embedding_dim': embedding_dim,
            'sequence_axis': sequence_axis,
            'feature_axis': feature_axis,
            'activation': activation,
            'output': output,}
        # successive dimensions of the dividing units
        __token_dim = [token_dim] if isinstance(token_dim, int) else token_dim
        # binary vs categorical probabilities
        __activation = 'softmax' if output == 'categorical' else 'sigmoid'
        # layers
        __layers = [
            # (B * G ^ i, E) => (B * G ^ (i+1), E)
            tokun.layers.DetokenizeBlock(
                sequence_axis=sequence_axis,
                feature_axis=feature_axis,
                token_dim=__g,
                embedding_dim=embedding_dim,
                activation=activation,
                name='detokenize-{}_{}'.format(__g, __i))
            for __i, __g in enumerate(__token_dim)] + [
            # (B * G ^ D, E) => (B * G ^ D, U)
            tokun.layers.HeadBlock(encoding_dim=encoding_dim, activation=__activation, name='project-head')]
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
        input_dim: int,
        output_dim: int,
        embedding_dim: int,
        sequence_axis: int=1,
        feature_axis: int=-1,
        activation: str='gelu',
        output: str='categorical',
        **kwargs
    ) -> None:
        # init
        super(AutoEncoder, self).__init__(**kwargs)
        # config
        self._config = {
            'token_dim': token_dim,
            'input_dim': input_dim,
            'output_dim': output_dim,
            'embedding_dim': embedding_dim,
            'sequence_axis': sequence_axis,
            'feature_axis': feature_axis,
            'activation': activation,
            'output': output,}
        # layers
        self._encoder = Encoder(token_dim=token_dim, encoding_dim=input_dim, embedding_dim=embedding_dim, sequence_axis=sequence_axis, feature_axis=feature_axis, activation=activation)
        self._decoder = Decoder(token_dim=token_dim[::-1], encoding_dim=output_dim, embedding_dim=embedding_dim, sequence_axis=sequence_axis, feature_axis=feature_axis, activation=activation, output=output)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        return self._decoder(self._encoder(x))

    def get_config(self) -> dict:
        __config = super(AutoEncoder, self).get_config()
        __config.update(self._config)
        return __config

    @classmethod
    def from_config(cls, config) -> tf.keras.layers.Layer:
        return cls(**config)
