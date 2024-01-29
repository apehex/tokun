import tensorflow as tf

import mlable.tensorflow.layers as _mtl

# FEED FORWARD BLOCK ##########################################################

class FeedForwardResidualBlock(tf.keras.Model):
    def __init__(
        self,
        norm_momentum=0.99,
        norm_epsilon=0.001,
        **kwargs
    ):
        super(FeedForwardResidualBlock, self).__init__(**kwargs)
        self._normalization = _mtl.LayerNormalization(axis=-1, momentum=norm_momentum, epsilon=norm_epsilon)
        self._projection = None

    def build(self, shape: tuple) -> None:
        # create the projection layer to matche the input shape
        self._projection = _mtl.Dense(units=shape[-1], use_bias=False)
        # build
        self._normalization.build(shape=shape)
        self._projection.build(shape=shape)
        # notify the model
        self.built = True

    def call(self, inputs: tf.Tensor, training: bool=True, **kwargs):
        __dx = inputs
        # normalize the features
        __dx = self._normalization(__dx, training=training, **kwargs)
        # projection: match the input shape
        __dx = self._projection(__dx, training=training, **kwargs)
        # residual
        return inputs + __dx

# ATTENTION BLOCK #############################################################

class ResidualSelfAttentionBlock(tf.keras.Model):
    def __init__(
        self,
        attention_head_dim: int,
        attention_head_count: int=1,
        norm_momentum=0.99,
        norm_epsilon=0.001,
        **kwargs
    ):
        super(ResidualSelfAttentionBlock, self).__init__(**kwargs)
        self._normalization = _mtl.LayerNormalization(axis=-1, momentum=norm_momentum, epsilon=norm_epsilon)
        self._attention = _mtl.Attention(head_dim=attention_head_dim, head_count=attention_head_count)
        self._projection = None

    def build(self, shape: tuple) -> None:
        # create the projection layer to matche the input shape
        self._projection = _mtl.Dense(units=shape[-1], use_bias=False)
        # build
        self._normalization.build(shape=shape)
        self._attention.build(shape=shape)
        self._projection.build(shape=list(shape)[:-1] + [self._attention._head_dim])
        # notify the model
        self.built = True

    def call(self, inputs: tf.Tensor, training: bool=True, **kwargs):
        __dx = inputs
        # normalize the features
        __dx = self._normalization(__dx, training=training, **kwargs)
        # self-attention
        __dx = self._attention(__dx, training=training, **kwargs)
        # projection: match the input shape
        __dx = self._projection(__dx, training=training, **kwargs)
        # residual
        return inputs + __dx
