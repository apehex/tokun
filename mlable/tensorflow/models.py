import tensorflow as tf

import mlable.tensorflow.layers as _mtl

# FEED FORWARD BLOCK ##########################################################

class ResidualFeedForwardBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        hidden_dim: int,
        norm_epsilon: float=0.001,
        **kwargs
    ):
        super(ResidualFeedForwardBlock, self).__init__(**kwargs)
        self._normalization = _mtl.LayerNormalization(axis=-1, epsilon=norm_epsilon)
        self._hidden_dim = hidden_dim
        self._hidden = _mtl.Dense(units=self._hidden_dim, use_bias=True)
        self._activation = _mtl.Activation(function=tf.math.tanh)
        self._projection = None

    def build(self, shape: tuple) -> None:
        # create the projection layer to match the input shape
        self._projection = _mtl.Dense(units=shape[-1], use_bias=True)
        # no need to build the activation layer
        self._normalization.build(shape=shape)
        self._hidden.build(shape=shape)
        self._projection.build(shape=list(shape)[:-1] + [self._hidden_dim])
        # notify the model
        self.built = True

    def call(self, inputs: tf.Tensor, **kwargs):
        __dx = inputs
        # normalize the features
        __dx = self._normalization(__dx, **kwargs)
        # expand inside the hidden layer
        __dx = self._hidden(__dx, **kwargs)
        __dx = self._activation(__dx, **kwargs)
        # projection: match the input shape
        __dx = self._projection(__dx, **kwargs)
        # residual
        return inputs + __dx

# ATTENTION BLOCK #############################################################

class ResidualSelfAttentionBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        attention_head_dim: int,
        attention_head_count: int=1,
        norm_epsilon=0.001,
        **kwargs
    ):
        super(ResidualSelfAttentionBlock, self).__init__(**kwargs)
        self._normalization = _mtl.LayerNormalization(axis=-1, epsilon=norm_epsilon)
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

    def call(self, inputs: tf.Tensor, **kwargs):
        __dx = inputs
        # normalize the features
        __dx = self._normalization(__dx, **kwargs)
        # self-attention
        __dx = self._attention(__dx, **kwargs)
        # projection: match the input shape
        __dx = self._projection(__dx, **kwargs)
        # residual
        return inputs + __dx

# META BLOCK ##################################################################

class ResidualSelfAttentionDecoderBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        hidden_dim: int,
        attention_head_dim: int,
        attention_head_count: int=1,
        norm_epsilon=0.001,
        **kwargs
    ):
        super(ResidualSelfAttentionDecoderBlock, self).__init__(**kwargs)
        self._feedforward = ResidualFeedForwardBlock(hidden_dim=hidden_dim, norm_epsilon=norm_epsilon)
        self._attention = ResidualSelfAttentionBlock(attention_head_dim=attention_head_dim, attention_head_count=attention_head_count, norm_epsilon=norm_epsilon)

    def build(self, shape: tuple) -> None:
        self._feedforward.build(shape=shape)
        self._attention.build(shape=shape)
        # notify the model
        self.built = True

    def call(self, inputs: tf.Tensor, **kwargs):
        __dx = inputs
        # self-attention
        __dx = self._attention(__dx, **kwargs)
        # projection: match the input shape
        __dx = self._feedforward(__dx, **kwargs)
        # residual
        return __dx
