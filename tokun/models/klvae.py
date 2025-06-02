"""Piece together the actual VAE CNN model for tokun."""

import functools

import keras
import tensorflow as tf

import mlable.blocks.convolution.resnet
import mlable.layers.shaping
import mlable.models.autoencoder

# CONSTANTS ###################################################################

DROPOUT = 0.0
EPSILON = 1e-6

# ENCODER #####################################################################

@keras.saving.register_keras_serializable(package='models')
class Encoder(tf.keras.models.Model):
    def __init__(
        self,
        channel_dim: iter,
        group_dim: int,
        head_dim: int,
        embed_dim: int,
        input_dim: int=256,
        layer_num: int=1,
        trainable: bool=True,
        dropout_rate: float=DROPOUT,
        epsilon_rate: float=EPSILON,
        **kwargs
    ) -> None:
        # init
        super(Encoder, self).__init__(**kwargs)
        # config
        self._config = {
            'channel_dim': [channel_dim] if isinstance(channel_dim, int) else list(channel_dim),
            'group_dim': max(1, group_dim),
            'head_dim': max(1, head_dim),
            'embed_dim': max(1, embed_dim),
            'input_dim': max(1, input_dim),
            'layer_num': max(1, layer_num),
            'dropout_rate': max(0.0, dropout_rate),
            'epsilon_rate': max(1e-8, epsilon_rate),
            'trainable': trainable,}
        # layers
        self._layers = []

    def build(self, input_shape: tuple) -> None:
        # parse
        __shape = tuple(input_shape)
        __count = len(self._config['channel_dim'])
        # factor
        __conv_args = {'kernel_size': 3, 'strides': 1, 'padding': 'same', 'data_format': 'channels_last',}
        __norm_args = {'axis': -1, 'epsilon': self._config['epsilon_rate'], 'center': True, 'scale': True,}
        __resnet_args = {'group_dim': self._config['group_dim'], 'layer_num': self._config['layer_num'], 'dropout_rate': self._config['dropout_rate'], 'epsilon_rate': self._config['epsilon_rate'],}
        # init
        self._layers = [
            # embed (B, H, W, C) => (B, H, W, C, E)
            tf.keras.layers.Embedding(
                input_dim=self._config['input_dim'],
                output_dim=self._config['embed_dim'],
                name='encoder-0-embed-bytes'),
            # merge (B, H, W, C, E) => (B, H, W, CE)
            mlable.layers.shaping.Merge(
                axis=-1,
                right=False,
                name='encoder-1-merge-embeds'),
            # even (B, H, W, CE) => (B, H, W, L0)
            tf.keras.layers.Conv2D(
                filters=self._config['channel_dim'][0],
                name='encoder-2-conv-inputs',
                **__conv_args),] + [
            # compress (B, H/2^i, W/2^i, Li) => (B, H/2^i+1, W/2^i+1, Li+1)
            mlable.blocks.convolution.resnet.EncoderBlock(
                channel_dim=__c,
                downsample_on=__i < (__count - 1),
                name=f'encoder-{__i + 3}-resnet-down',
                **__resnet_args)
            for __i, __c in enumerate(self._config['channel_dim'])] + [
            # transform (B, Hn, Wn, Ln) => (B, Hn, Wn, Ln)
            mlable.blocks.convolution.resnet.TransformerBlock(
                channel_dim=self._config['channel_dim'][-1],
                head_dim=self._config['head_dim'],
                use_causal_mask=False,
                name=f'encoder-{__count + 4}-resnet-mid',
                **__resnet_args),
            # normalize (B, Hn, Wn, Ln) => (B, Hn, Wn, Ln)
            tf.keras.layers.GroupNormalization(
                groups=self._config['group_dim'],
                name=f'encoder-{__count + 5}-norm',
                **__norm_args),
            # activation (B, Hn, Wn, Ln) => (B, Hn, Wn, Ln)
            tf.keras.layers.Activation(
                activation='silu',
                name=f'encoder-{__count + 6}-activation'),
            # expand (B, Hn, Wn, Ln) => (B, Hn, Wn, 2Ln)
            tf.keras.layers.Conv2D(
                filters=2 * self._config['channel_dim'][-1],
                name=f'encoder-{__count + 7}-conv-outputs',
                **__conv_args),
            ]
        # build
        for __l in self._layers:
            __l.build(__shape)
            __shape = __l.compute_output_shape(__shape)
        # freeze
        for __l in self._layers:
            __l.trainable = self._config['trainable']
        # register
        self.built = True

    def compute_output_shape(self, input_shape: tuple) -> tuple:
        return functools.reduce(lambda __s, __l: __l.compute_output_shape(__s), self._layers, input_shape)

    def call(self, inputs: tf.Tensor, training: bool=False, **kwargs) -> tf.Tensor:
        return functools.reduce(lambda __x, __l: __l(__x, training=training, **kwargs), self._layers, inputs)

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
        channel_dim: iter,
        group_dim: int,
        head_dim: int,
        output_dim: int,
        layer_num: int=1,
        trainable: bool=True,
        dropout_rate: float=DROPOUT,
        epsilon_rate: float=EPSILON,
        **kwargs
    ) -> None:
        # init
        super(Decoder, self).__init__(**kwargs)
        # config
        self._config = {
            'channel_dim': [channel_dim] if isinstance(channel_dim, int) else list(channel_dim),
            'group_dim': max(1, group_dim),
            'head_dim': max(1, head_dim),
            'output_dim': max(1, output_dim),
            'layer_num': max(1, layer_num),
            'dropout_rate': max(0.0, dropout_rate),
            'epsilon_rate': max(1e-8, epsilon_rate),
            'trainable': trainable,}
        # layers
        self._layers = []

    def build(self, input_shape: tuple) -> None:
        # parse
        __shape = tuple(input_shape)
        __count = len(self._config['channel_dim'])
        # factor
        __conv_args = {'kernel_size': 3, 'strides': 1, 'padding': 'same', 'data_format': 'channels_last',}
        __norm_args = {'axis': -1, 'epsilon': self._config['epsilon_rate'], 'center': True, 'scale': True,}
        __resnet_args = {'group_dim': self._config['group_dim'], 'layer_num': self._config['layer_num'], 'dropout_rate': self._config['dropout_rate'], 'epsilon_rate': self._config['epsilon_rate'],}
        # init
        self._layers = [
            # transform (B, Hn, Wn, Ln) => (B, Hn, Wn, Ln)
            mlable.blocks.convolution.resnet.TransformerBlock(
                channel_dim=self._config['channel_dim'][0],
                head_dim=self._config['head_dim'],
                use_causal_mask=False,
                name=f'decoder-0-resnet-mid',
                **__resnet_args),] + [
            # decompress (B, H/2^i+1, W/2^i+1, Li+1) => (B, H/2^i, W/2^i, Li)
            mlable.blocks.convolution.resnet.DecoderBlock(
                channel_dim=__c,
                upsample_on=__i < (__count - 1),
                name=f'decoder-{__i + 1}-resnet-up',
                **__resnet_args)
            for __i, __c in enumerate(self._config['channel_dim'])] + [
            # normalize (B, H, W, L0) => (B, H, W, L0)
            tf.keras.layers.GroupNormalization(
                groups=self._config['group_dim'],
                name=f'decoder-{__count + 2}-norm',
                **__norm_args),
            # activation (B, H, W, L0) => (B, H, W, L0)
            tf.keras.layers.Activation(
                activation='silu',
                name=f'decoder-{__count + 3}-activation'),
            # expand (B, H, W, L0) => (B, H, W, O) where O = 8C typically
            tf.keras.layers.Conv2D(
                filters=self._config['output_dim'],
                name=f'decoder-{__count + 4}-conv-outputs',
                **__conv_args),
            ]
        # build
        for __l in self._layers:
            __l.build(__shape)
            __shape = __l.compute_output_shape(__shape)
        # freeze
        for __l in self._layers:
            __l.trainable = self._config['trainable']
        # register
        self.built = True

    def compute_output_shape(self, input_shape: tuple) -> tuple:
        return functools.reduce(lambda __s, __l: __l.compute_output_shape(__s), self._layers, input_shape)

    def call(self, inputs: tf.Tensor, training: bool=False, **kwargs) -> tf.Tensor:
        return functools.reduce(lambda __x, __l: __l(__x, training=training, **kwargs), self._layers, inputs)

    def get_config(self) -> dict:
        __config = super(Decoder, self).get_config()
        __config.update(self._config)
        return __config

    @classmethod
    def from_config(cls, config) -> tf.keras.layers.Layer:
        return cls(**config)
