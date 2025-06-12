"""Piece together the actual VAE CNN model for tokun."""

import functools

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

@tf.keras.utils.register_keras_serializable(package='models')
class Encoder(tf.keras.models.Model):
    def __init__(
        self,
        block_cfg: iter,
        embed_dim: int,
        input_dim: int=256,
        dropout_rate: float=DROPOUT,
        epsilon_rate: float=EPSILON,
        **kwargs
    ) -> None:
        super(Encoder, self).__init__(**kwargs)
        # save for IO serialization
        self._config = {
            'block_cfg': block_cfg,
            'input_dim': max(1, input_dim),
            'embed_dim': max(1, embed_dim),
            'dropout_rate': max(0.0, dropout_rate),
            'epsilon_rate': max(1e-8, epsilon_rate),}
        # layers
        self._embed_byte = None
        self._merge_bytes = None
        self._embed_height = None
        self._embed_width = None
        self._unet_blocks = []

    def build(self, input_shape: tuple) -> None:
        __shape = tuple(input_shape)
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
        # compress (B, H/2^i, W/2^i, Li) => (B, H/2^i+1, W/2^i+1, Li+1)
        self._unet_blocks = [
            mlable.blocks.convolution.unet.UnetBlock(
                name=f'encoder-{__i + 4}-unet-down',
                **__c)
            for __i, __c in enumerate(self.get_unet_configs())]
        # build
        for __b in self.get_all_blocks():
            __b.build(__shape)
            __shape = __b.compute_output_shape(__shape)
        # register
        self.built = True

    def call(self, inputs: tf.Tensor, training: bool=False, **kwargs) -> tf.Tensor:
        return functools.reduce(lambda __x, __b: __b(__x, training=training, **kwargs), self.get_all_blocks(), inputs)

    def compute_output_shape(self, input_shape: tuple) -> tuple:
        return functools.reduce(lambda __s, __b: __b.compute_output_shape(__s), self.get_all_blocks(), input_shape)

    def get_all_blocks(self) -> list:
        return [self._embed_byte, self._merge_bytes, self._embed_height, self._embed_width] + self._unet_blocks

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

    def get_unet_configs(self) -> dict:
        return [
            {
                'channel_dim': self._config['block_cfg'][0]['channel_dim'],
                'group_dim': None,
                'head_dim': None,
                'head_num': None,
                'layer_num': 1,
                'add_attention': False,
                'add_downsampling': False,
                'add_upsampling': False,
                'dropout_rate': self._config['dropout_rate'],
                'epsilon_rate': self._config['epsilon_rate'],}] + [
            {
                'channel_dim': __c['channel_dim'],
                'group_dim': __c.get('group_dim', None),
                'head_dim': __c.get('head_dim', None),
                'head_num': __c.get('head_num', None),
                'layer_num': __c.get('layer_num', 2),
                'add_attention': __c.get('add_attention', False),
                'add_downsampling': __c.get('add_downsampling', False),
                'add_upsampling': __c.get('add_upsampling', False),
                'dropout_rate': self._config['dropout_rate'],
                'epsilon_rate': self._config['epsilon_rate'],}
            for __c in self._config['block_cfg']] + [
            {
                'channel_dim': 2 * self._config['block_cfg'][-1]['channel_dim'],
                'group_dim': None,
                'head_dim': None,
                'head_num': None,
                'layer_num': 1,
                'add_attention': False,
                'add_downsampling': False,
                'add_upsampling': False,
                'dropout_rate': self._config['dropout_rate'],
                'epsilon_rate': self._config['epsilon_rate'],}]

    @classmethod
    def from_config(cls, config) -> tf.keras.layers.Layer:
        return cls(**config)

# DECODER #####################################################################

@tf.keras.utils.register_keras_serializable(package='models')
class Decoder(tf.keras.models.Model):
    def __init__(
        self,
        block_cfg: iter,
        output_dim: int,
        dropout_rate: float=DROPOUT,
        epsilon_rate: float=EPSILON,
        **kwargs
    ) -> None:
        super(Decoder, self).__init__(**kwargs)
        # save for IO serialization
        self._config = {
            'block_cfg': block_cfg,
            'output_dim': max(1, output_dim),
            'dropout_rate': max(0.0, dropout_rate),
            'epsilon_rate': max(1e-8, epsilon_rate),}
        # blocks
        self._unet_blocks = []

    def build(self, input_shape: tuple) -> None:
        __shape = tuple(input_shape)
        # decompress (B, H/2^i+1, W/2^i+1, Li+1) => (B, H/2^i, W/2^i, Li)
        self._unet_blocks = [
            mlable.blocks.convolution.unet.UnetBlock(
                name=f'decoder-{__i}-unet-up',
                **__c)
            for __i, __c in enumerate(self.get_unet_configs())]
        # build
        for __b in self._unet_blocks:
            __b.build(__shape)
            __shape = __b.compute_output_shape(__shape)
        # register
        self.built = True

    def compute_output_shape(self, input_shape: tuple) -> tuple:
        return functools.reduce(lambda __s, __b: __b.compute_output_shape(__s), self._unet_blocks, input_shape)

    def call(self, inputs: tf.Tensor, training: bool=False, **kwargs) -> tf.Tensor:
        return functools.reduce(lambda __x, __b: __b(__x, training=training, **kwargs), self._unet_blocks, inputs)

    def get_config(self) -> dict:
        __config = super(Decoder, self).get_config()
        __config.update(self._config)
        return __config

    def get_unet_configs(self) -> dict:
        return [
            {
                'channel_dim': __c['channel_dim'],
                'group_dim': __c.get('group_dim', None),
                'head_dim': __c.get('head_dim', None),
                'head_num': __c.get('head_num', None),
                'layer_num': __c.get('layer_num', 2),
                'add_attention': __c.get('add_attention', False),
                'add_downsampling': __c.get('add_downsampling', False),
                'add_upsampling': __c.get('add_upsampling', False),
                'dropout_rate': self._config['dropout_rate'],
                'epsilon_rate': self._config['epsilon_rate'],}
            for __c in self._config['block_cfg']] + [
            {
                'channel_dim': self._config['output_dim'],
                'group_dim': None,
                'head_dim': None,
                'head_num': None,
                'layer_num': 2,
                'add_attention': False,
                'add_downsampling': False,
                'add_upsampling': False,
                'dropout_rate': self._config['dropout_rate'],
                'epsilon_rate': self._config['epsilon_rate'],}]

    @classmethod
    def from_config(cls, config) -> tf.keras.layers.Layer:
        return cls(**config)

# VAE #########################################################################

@tf.keras.utils.register_keras_serializable(package='models')
class KlAutoEncoder(mlable.models.autoencoder.VaeModel):
    def __init__(
        self,
        block_cfg: iter,
        embed_dim: int,
        output_dim: int,
        input_dim: int=256,
        step_min: int=0,
        step_max: int=2 ** 12,
        beta_min: float=0.0,
        beta_max: float=1.0,
        dropout_rate: float=DROPOUT,
        epsilon_rate: float=EPSILON,
        **kwargs
    ) -> None:
        super(KlAutoEncoder, self).__init__(step_min=step_min, step_max=step_max, beta_min=beta_min, beta_max=beta_max, **kwargs)
        # save for IO serialization
        self._config.update({
            'block_cfg': block_cfg,
            'input_dim': max(1, input_dim),
            'embed_dim': max(1, embed_dim),
            'output_dim': max(1, output_dim),
            'dropout_rate': max(0.0, dropout_rate),
            'epsilon_rate': max(1e-8, epsilon_rate),})
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
            'block_cfg': [
                {
                    'add_downsampling' if ('sampling' in __k) else __k: __v
                    for __k, __v in __c.items()}
                for __c in self._config['block_cfg']],
            **{__k: self._config[__k] for __k in ['input_dim', 'embed_dim', 'dropout_rate', 'epsilon_rate']}}

    def get_decoder_config(self) -> dict:
        return {
            'block_cfg': [
                {
                    'add_upsampling' if ('sampling' in __k) else __k: __v
                    for __k, __v in __c.items()}
                for __c in reversed(self._config['block_cfg'])],
            **{__k: self._config[__k] for __k in ['output_dim', 'dropout_rate', 'epsilon_rate']}}

    @classmethod
    def from_config(cls, config) -> tf.keras.layers.Layer:
        return cls(**config)
