import tensorflow as tf

import mlable.tensorflow.initializers as _mti

# NORMALIZATION ###############################################################

class BatchNormalization(tf.keras.layers.Layer):
    def __init__(
        self,
        axis=0,
        momentum=0.99,
        epsilon=0.001,
        **kwargs
    ):
        super(BatchNormalization, self).__init__(**kwargs)
        self._axis = axis
        self._momentum = momentum
        self._epsilon = epsilon
        self._mean = None
        self._stddev = None
        self._gain = None
        self._bias = None

    def build(self, shape: tuple):
        # shape
        __axis = self._axis % len(shape) # positive index even when the axis is specified negatively, like -2
        __shape = [__d for __i, __d in enumerate(shape) if __i != __axis]
        # values
        __mean_init = _mti.SmallNormal()
        __stddev_init = _mti.SmallNormal()
        __gain_init = _mti.SmallNormal()
        __bias_init = _mti.SmallNormal()
        # tensors
        self._mean = self.add_weight("mean", shape=__shape, initializer=__mean_init)
        self._stddev = self.add_weight("stddev", shape=__shape, initializer=__stddev_init)
        self._gain = self.add_weight("gain", shape=__shape, initializer=__gain_init)
        self._bias = self.add_weight("bias", shape=__shape, initializer=__bias_init)

    def call(self, inputs: tf.Tensor, training: bool=True, **kwargs):
        if training:
            # current values
            __batch_mean = tf.math.reduce_mean(inputs, axis=self._axis, keepdims=True)
            __batch_stddev = tf.math.reduce_std(inputs, axis=self._axis, keepdims=True)
            # update parameters
            self._mean = tf.stop_gradient(self._momentum * self._mean + (1. - self._momentum) * __batch_mean)
            self._stddev = tf.stop_gradient(self._momentum * self._stddev + (1. - self._momentum) * __batch_stddev)
        # normalize
        __normalized = tf.math.divide(inputs - self._mean, self._stddev + self._epsilon)
        # scale
        return tf.math.multiply(self._gain, __normalized) + self._bias

class LayerNormalization(tf.keras.layers.Layer):
    def __init__(
        self,
        axis=-1,
        momentum=0.99,
        epsilon=0.001,
        **kwargs
    ):
        super(BatchNormalization, self).__init__(**kwargs)
        self._axis = axis
        self._momentum = momentum
        self._epsilon = epsilon
        self._mean = None
        self._stddev = None
        self._gain = None
        self._bias = None

    def build(self, shape: tuple):
        # shape
        __axis = self._axis % len(shape) # positive index even when the axis is specified negatively, like -2
        __shape = [__d for __i, __d in enumerate(shape) if __i != __axis]
        # values
        __mean_init = _mti.SmallNormal()
        __stddev_init = _mti.SmallNormal()
        __gain_init = _mti.SmallNormal()
        __bias_init = _mti.SmallNormal()
        # tensors
        self._mean = self.add_weight("mean", shape=__shape, initializer=__mean_init)
        self._stddev = self.add_weight("stddev", shape=__shape, initializer=__stddev_init)
        self._gain = self.add_weight("gain", shape=__shape, initializer=__gain_init)
        self._bias = self.add_weight("bias", shape=__shape, initializer=__bias_init)

    def call(self, inputs: tf.Tensor, training: bool=True, **kwargs):
        if training:
            # current values
            __layer_mean = tf.math.reduce_mean(inputs, axis=self._axis, keepdims=True)
            __layer_stddev = tf.math.reduce_std(inputs, axis=self._axis, keepdims=True)
            # update parameters
            self._mean = tf.stop_gradient(self._momentum * self._mean + (1. - self._momentum) * __layer_mean)
            self._stddev = tf.stop_gradient(self._momentum * self._stddev + (1. - self._momentum) * __layer_stddev)
        # normalize
        __normalized = tf.math.divide(inputs - self._mean, self._stddev + self._epsilon)
        # scale
        return tf.math.multiply(self._gain, __normalized) + self._bias

# REGULARIZATION ##############################################################

# dropout

# LINEAR ######################################################################

class Dense(tf.keras.layers.Layer):
    def __init__(
        self,
        units: int,
        use_bias: bool=True,
        **kwargs
    ):
        super(Dense, self).__init__(**kwargs)
        self._units = units
        self._biased = use_bias
        self._kernel = None
        self._bias = None

    def build(self, shape: tuple):
        # kernel
        __kernel_init = _mti.SmallNormal()
        self._kernel = self.add_weight("kernel", shape=[int(shape[-1]), self._units], initializer=__kernel_init)
        # bias
        if self._biased:
            __bias_init = _mti.SmallNormal()
            self._bias = self.add_weight("bias", shape=[self._units], initializer=__bias_init)

    def call(self, inputs: tf.Tensor, **kwargs):
        return tf.matmul(inputs, self._kernel) + self._bias if (self._biased and self._bias is not None) else tf.matmul(inputs, self._kernel)

# QUADRATIC ###################################################################

class Attention(tf.keras.layers.Layer):
    def __init__(
        self,
        dropout: float,
        **kwargs
    ):
        super(Attention, self).__init__(**kwargs)

    def build(self, shape: tuple):
        # kernel
        __kernel_init = _mti.SmallNormal()
        self._kernel = self.add_weight("kernel", shape=[int(shape[-1]), self._units], initializer=__kernel_init)

    def call(self, inputs: tf.Tensor, **kwargs):
        # key
        # query
        # value
        # score
        return self._function(inputs)

# EMBEDDING ###################################################################

class Embedding(Dense):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        **kwargs
    ):
        super(Embedding, self).__init__(units=output_dim, use_bias=False, **kwargs)
        self._input_dim = input_dim
        self._output_dim = output_dim

    def build(self, shape: tuple):
        __shape = list(shape)
        # add a direction for the one-hot encoding
        __shape = __shape + [self._input_dim]
        # init
        super(Embedding, self).build(shape=__shape)

    def call(self, inputs: tf.Tensor, **kwargs):
        __x = tf.one_hot(indices=inputs, depth=self._input_dim, dtype=tf.dtypes.float32)
        return super(Embedding, self).call(inputs=__x, **kwargs)

class PositionEmbedding(Dense):
    def __init__(
        self,
        output_dim: int,
        **kwargs
    ):
        super(PositionEmbedding, self).__init__(units=output_dim, use_bias=False, **kwargs)
        self._time_dim = None

    def build(self, shape: tuple):
        # save the time dim
        self._time_dim = list(shape)[-1]
        # init
        super(PositionEmbedding, self).build(shape=shape)

    def call(self, inputs: tf.Tensor, **kwargs):
        __p = 0.
        return super(PositionEmbedding, self).call(inputs=__x, **kwargs)

# RESIDUALS ###################################################################

# ACTIVATION ##################################################################

class Activation(tf.keras.layers.Layer):
    def __init__(
        self,
        function: callable,
        **kwargs
    ):
        super(Activation, self).__init__(**kwargs)
        self._function = function

    def call(self, inputs: tf.Tensor, **kwargs):
        return self._function(inputs)

class Softmax(tf.keras.layers.Layer):
    def __init__(
        self,
        axis: int=-1,
        **kwargs
    ):
        super(Softmax, self).__init__(**kwargs)
        self._axis = axis

    def call(self, inputs: tf.Tensor, **kwargs):
        return tf.nn.softmax(inputs, axis=self._axis)

# RESHAPING ###################################################################

class Merge(tf.keras.layers.Layer):
    def __init__(
        self,
        axis: int,
        n: int,
        **kwargs
    ):
        super(Merge, self).__init__(**kwargs)
        self._axis = axis
        self._n = n

    def call(self, inputs: tf.Tensor, **kwargs):
        __shape = list(inputs.shape)
        __axis0 = self._axis % len(__shape)
        __axis1 = (self._axis + 1) % len(__shape)
        # merge n rows along the given axis
        __shape[__axis0] = inputs.shape[__axis0] // self._n
        __shape[__axis1] = inputs.shape[__axis1] * self._n
        return tf.squeeze(tf.reshape(inputs, __shape))

class Reshape(tf.keras.layers.Layer):
    def __init__(
        self,
        target_shape: tuple,
        **kwargs
    ):
        super(Reshape, self).__init__(**kwargs)
        self._shape = target_shape

    def call(self, inputs: tf.Tensor, **kwargs):
        return tf.reshape(inputs, self._shape)
