"""Piece together the actual VAE CNN model for tokun."""

import functools

import keras
import tensorflow as tf

import mlable.blocks.convolution.resnet
import mlable.blocks.convolution.unet
import mlable.layers.embedding
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
            'group_dim': group_dim,
            'head_dim': head_dim,
            'embed_dim': max(1, embed_dim),
            'input_dim': max(1, input_dim),
            'layer_num': max(1, layer_num),
            'dropout_rate': max(0.0, dropout_rate),
            'epsilon_rate': max(1e-8, epsilon_rate),
            'trainable': trainable,}
        # layers
        self._embed_byte = None
        self._merge_bytes = None
        self._embed_height = None
        self._embed_width = None
        self._resnet_preprocess = None
        self._resnet_postprocess = None
        self._unet_blocks = []

    def build(self, input_shape: tuple) -> None:
        # parse
        __shape = tuple(input_shape)
        __count = len(self._config['channel_dim'])
        # embed (B, H, W, C) => (B, H, W, C, E)
        self._embed_byte = tf.keras.layers.Embedding(
            name='encoder-0-embed-bytes',
            **self.get_embed_byte_config())
        # merge (B, H, W, C, E) => (B, H, W, CE)
        self._merge_bytes = mlable.layers.shaping.Merge(
            name='encoder-1-merge-embeds',
            **self.get_merge_config())
        # embed height (B, H, W, C) => (B, H, W, CE)
        self._embed_height = mlable.layers.embedding.PositionalEmbedding(
            name='encoder-2-embed-height',
            **self.get_embed_height_config())
        # embed width (B, H, W, C) => (B, H, W, CE)
        self._embed_width = mlable.layers.embedding.PositionalEmbedding(
            name='encoder-3-embed-width',
            **self.get_embed_width_config())
        # even (B, H, W, CE) => (B, H, W, L0)
        self._resnet_preprocess = mlable.blocks.convolution.resnet.ResnetBlock(
            channel_dim=self._config['channel_dim'][0],
            name='encoder-4-resnet-preprocess',
            **self.get_resnet_config())
        # expand (B, Hn, Wn, Ln) => (B, Hn, Wn, 2Ln)
        self._resnet_postprocess = mlable.blocks.convolution.resnet.ResnetBlock(
            channel_dim=2 * self._config['channel_dim'][-1],
            name=f'encoder-{__count + 5}-resnet-postprocess',
            **self.get_resnet_config())
        # compress (B, H/2^i, W/2^i, Li) => (B, H/2^i+1, W/2^i+1, Li+1)
        self._unet_blocks = [
            mlable.blocks.convolution.unet.UnetBlock(
                channel_dim=__c,
                add_attention=__i == (__count - 1),
                add_downsampling=__i < (__count - 1),
                add_upsampling=False,
                name=f'encoder-{__i + 5}-unet-down',
                **self.get_unet_config())
            for __i, __c in enumerate(self._config['channel_dim'])]
        # build
        for __b in self.get_all_blocks():
            __b.build(__shape)
            __shape = __b.compute_output_shape(__shape)
        # freeze
        for __b in self.get_all_blocks():
            __b.trainable = self._config['trainable']
        # register
        self.built = True

    def call(self, inputs: tf.Tensor, training: bool=False, **kwargs) -> tf.Tensor:
        return functools.reduce(lambda __x, __b: __b(__x, training=training, **kwargs), self.get_all_blocks(), inputs)

    def compute_output_shape(self, input_shape: tuple) -> tuple:
        return functools.reduce(lambda __s, __b: __b.compute_output_shape(__s), self.get_all_blocks(), input_shape)

    def get_embed_blocks(self) -> list:
        return [self._embed_byte, self._merge_bytes, self._embed_height, self._embed_width]

    def get_resnet_blocks(self) -> list:
        return [self._resnet_preprocess] + self._unet_blocks + [self._resnet_postprocess]

    def get_all_blocks(self) -> list:
        return self.get_embed_blocks() + self.get_resnet_blocks()

    def get_config(self) -> dict:
        __config = super(Encoder, self).get_config()
        __config.update(self._config)
        return __config

    def get_embed_byte_config(self) -> dict:
        return {'input_dim': self._config['input_dim'], 'output_dim': self._config['embed_dim'],}

    def get_embed_height_config(self) -> dict:
        return {'sequence_axis': 1, 'feature_axis': -1,}

    def get_embed_width_config(self) -> dict:
        return {'sequence_axis': 2, 'feature_axis': -1,}

    def get_merge_config(self) -> dict:
        return {'axis': -1, 'right': False,}

    def get_resnet_config(self) -> dict:
        return {
            __k: self._config[__k]
            for __k in ['group_dim', 'dropout_rate', 'epsilon_rate']}

    def get_unet_config(self) -> dict:
        return {
            __k: self._config[__k]
            for __k in ['group_dim', 'head_dim', 'layer_num', 'dropout_rate', 'epsilon_rate']}

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
            'group_dim': group_dim,
            'head_dim': head_dim,
            'output_dim': max(1, output_dim),
            'layer_num': max(1, layer_num),
            'dropout_rate': max(0.0, dropout_rate),
            'epsilon_rate': max(1e-8, epsilon_rate),
            'trainable': trainable,}
        # blocks
        self._unet_blocks = []
        self._resnet_project = None

    def build(self, input_shape: tuple) -> None:
        # parse
        __shape = tuple(input_shape)
        __count = len(self._config['channel_dim'])
        # compress (B, H/2^i+1, W/2^i+1, Li+1) => (B, H/2^i, W/2^i, Li)
        self._unet_blocks = [
            mlable.blocks.convolution.unet.UnetBlock(
                channel_dim=__c,
                add_attention=__i == 0,
                add_downsampling=False,
                add_upsampling=__i > 0,
                name=f'decoder-{__i}-unet-up',
                **self.get_unet_config())
            for __i, __c in enumerate(self._config['channel_dim'])]
        # project (B, H, W, L0) => (B, H, W, O)
        self._resnet_project = mlable.blocks.convolution.resnet.ResnetBlock(
            channel_dim=self._config['output_dim'],
            name=f'decoder-{__count}-resnet-project',
            **self.get_resnet_config())
        # build
        for __b in self.get_all_blocks():
            __b.build(__shape)
            __shape = __b.compute_output_shape(__shape)
        # freeze
        for __b in self.get_all_blocks():
            __b.trainable = self._config['trainable']
        # register
        self.built = True

    def compute_output_shape(self, input_shape: tuple) -> tuple:
        return functools.reduce(lambda __s, __b: __b.compute_output_shape(__s), self.get_all_blocks(), input_shape)

    def call(self, inputs: tf.Tensor, training: bool=False, **kwargs) -> tf.Tensor:
        return functools.reduce(lambda __x, __b: __b(__x, training=training, **kwargs), self.get_all_blocks(), inputs)

    def get_all_blocks(self) -> list:
        return self._unet_blocks + [self._resnet_project]

    def get_config(self) -> dict:
        __config = super(Decoder, self).get_config()
        __config.update(self._config)
        return __config

    def get_resnet_config(self) -> dict:
        return {
            __k: self._config[__k]
            for __k in ['group_dim', 'dropout_rate', 'epsilon_rate']}

    def get_unet_config(self) -> dict:
        return {
            __k: self._config[__k]
            for __k in ['group_dim', 'head_dim', 'layer_num', 'dropout_rate', 'epsilon_rate']}

    @classmethod
    def from_config(cls, config) -> tf.keras.layers.Layer:
        return cls(**config)

# VAE #########################################################################

@keras.saving.register_keras_serializable(package='models')
class KlAutoEncoder(mlable.models.autoencoder.VaeModel):
    def __init__(
        self,
        channel_dim: iter,
        group_dim: int,
        head_dim: int,
        embed_dim: int,
        output_dim: int,
        input_dim: int=256,
        layer_num: int=2,
        step_min: int=0,
        step_max: int=2 ** 12,
        beta_min: float=0.0,
        beta_max: float=1.0,
        trainable: bool=True,
        dropout_rate: float=DROPOUT,
        epsilon_rate: float=EPSILON,
        **kwargs
    ) -> None:
        # init
        super(KlAutoEncoder, self).__init__(step_min=step_min, step_max=step_max, beta_min=beta_min, beta_max=beta_max, **kwargs)
        # config
        self._config.update({
            'channel_dim': [channel_dim] if isinstance(channel_dim, int) else list(channel_dim),
            'group_dim': group_dim,
            'head_dim': head_dim,
            'embed_dim': max(1, embed_dim),
            'output_dim': max(1, output_dim),
            'input_dim': max(1, input_dim),
            'layer_num': max(1, layer_num),
            'dropout_rate': max(0.0, dropout_rate),
            'epsilon_rate': max(1e-8, epsilon_rate),
            'trainable': trainable,})
        # layers
        self._encoder = None
        self._decoder = None

    def build(self, input_shape: tuple) -> None:
        # init
        self._encoder = Encoder(**self.get_encoder_config())
        self._decoder = Decoder(**self.get_decoder_config())
        # build
        __shape = tuple(input_shape)
        self._encoder.build(__shape)
        __shape = self.compute_latent_shape(__shape)
        self._decoder.build(__shape)
        # register
        self.built = True

    def encode(self, inputs: tf.Tensor, training: bool=False, **kwargs) -> tuple:
        __outputs = self._encoder(inputs, training=training, **kwargs)
        # split in 2: mean + logvar
        return tuple(tf.split(__outputs, num=2, num_or_size_splits=2, axis=-1))

    def decode(self, inputs: tf.Tensor, training: bool=False, **kwargs) -> tf.Tensor:
        return self._decoder(inputs, training=training, **kwargs)

    def compute_latent_shape(self, input_shape: tuple) -> tuple:
        __shape = self._encoder.compute_output_shape(input_shape)
        # split in 2, because the encoder returns both mean and logvar
        return tuple(__shape[:-1]) + (__shape[-1] // 2,)

    def compute_output_shape(self, input_shape: tuple) -> tuple:
        return self._decoder.compute_output_shape(self.compute_latent_shape(input_shape))

    def get_config(self) -> dict:
        __config = super(KlAutoEncoder, self).get_config()
        __config.update(self._config)
        return __config

    def get_encoder_config(self) -> dict:
        return {
            __k: self._config[__k]
            for __k in ['channel_dim', 'group_dim', 'embed_dim', 'input_dim', 'head_dim', 'layer_num', 'dropout_rate', 'epsilon_rate', 'trainable']}

    def get_decoder_config(self) -> dict:
        return {
            __k: list(reversed(self._config[__k])) if (__k == 'channel_dim') else self._config[__k]
            for __k in ['channel_dim', 'group_dim', 'output_dim', 'head_dim', 'layer_num', 'dropout_rate', 'epsilon_rate', 'trainable']}

    @classmethod
    def from_config(cls, config) -> tf.keras.layers.Layer:
        return cls(**config)
