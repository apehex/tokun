import tensorflow as tf

import mlable.tensorflow.layers as _mtl

# FEED FORWARD BLOCK ##########################################################

class ResidualFeedForwardBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        hidden_dim: int,
        normalization_epsilon: float=0.001,
        **kwargs
    ):
        super(ResidualFeedForwardBlock, self).__init__(**kwargs)
        self._normalization = _mtl.LayerNormalization(axis=-1, epsilon=normalization_epsilon)
        self._hidden_dim = hidden_dim
        self._hidden = _mtl.Dense(units=self._hidden_dim, use_bias=True)
        self._activation = _mtl.Activation(function=tf.math.tanh)
        self._projection = None

    def build(self, input_shape: tuple, **kwargs) -> None:
        # create the projection layer to match the input shape
        self._projection = _mtl.Dense(units=input_shape[-1], use_bias=True)
        # no need to build the activation layer
        self._normalization.build(input_shape=input_shape) # no weights
        self._hidden.build(input_shape=input_shape) # (C, H)
        self._projection.build(input_shape=list(input_shape)[:-1] + [self._hidden_dim]) # (H, C)
        # notify the model
        self.built = True

    def call(self, inputs: tf.Tensor, **kwargs):
        __dx = inputs # (B, T, C)
        # normalize the features
        __dx = self._normalization(__dx, **kwargs) # (B, T, C)
        # expand inside the hidden layer
        __dx = self._hidden(__dx, **kwargs) # (B, T, H) = (B, T, C) * (C, H)
        __dx = self._activation(__dx, **kwargs) # (B, T, H)
        # projection: match the input shape
        __dx = self._projection(__dx, **kwargs) # (B, T, C) = (B, T, H) * (H, C)
        # residual
        return inputs + __dx # (B, T, C)

# ATTENTION BLOCK #############################################################

class ResidualSelfAttentionBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        attention_head_dim: int,
        attention_head_count: int=1,
        normalization_epsilon: float=0.001,
        **kwargs
    ):
        super(ResidualSelfAttentionBlock, self).__init__(**kwargs)
        self._normalization = _mtl.LayerNormalization(axis=-1, epsilon=normalization_epsilon)
        self._attention = _mtl.Attention(head_dim=attention_head_dim, head_count=attention_head_count)
        self._projection = None

    def build(self, input_shape: tuple, **kwargs) -> None:
        # create the projection layer to matche the input shape
        self._projection = _mtl.Dense(units=input_shape[-1], use_bias=False)
        # build
        self._normalization.build(input_shape=input_shape)
        self._attention.build(input_shape=input_shape)
        self._projection.build(input_shape=list(input_shape)[:-1] + [self._attention._head_dim])
        # notify the model
        self.built = True

    def call(self, inputs: tf.Tensor, **kwargs):
        __dx = inputs # (B, T, C)
        # normalize the features
        __dx = self._normalization(__dx, **kwargs) # (B, T, C)
        # self-attention
        __dx = self._attention(__dx, **kwargs) # (B, T, H) = ((B, T, C) * (C, H)) * t((B, T, C) * (C, H)) * (T, H)
        # projection: match the input shape
        __dx = self._projection(__dx, **kwargs) # (B, T, C) = (B, T, H) * (H, H') * (H', H)
        # residual
        return inputs + __dx # (B, T, C)

# META BLOCK ##################################################################

class ResidualSelfAttentionDecoderBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        hidden_dim: int,
        attention_head_dim: int,
        attention_head_count: int=1,
        normalization_epsilon: float=0.001,
        **kwargs
    ):
        super(ResidualSelfAttentionDecoderBlock, self).__init__(**kwargs)
        self._feedforward = ResidualFeedForwardBlock(hidden_dim=hidden_dim, normalization_epsilon=normalization_epsilon)
        self._attention = ResidualSelfAttentionBlock(attention_head_dim=attention_head_dim, attention_head_count=attention_head_count, normalization_epsilon=normalization_epsilon)

    def build(self, input_shape: tuple, **kwargs) -> None:
        self._feedforward.build(input_shape=input_shape)
        self._attention.build(input_shape=input_shape)
        # notify the model
        self.built = True

    def call(self, inputs: tf.Tensor, **kwargs):
        __dx = inputs # (B, T, C)
        # residual self-attention
        __dx = self._attention(__dx, **kwargs) # (B, T, C)
        # residual FF
        __dx = self._feedforward(__dx, **kwargs) # (B, T, C)
        # residual
        return __dx # (B, T, C)
