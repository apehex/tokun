import numpy as np
import tensorflow as tf

import mlable.tensorflow.initializers as _mti

# NORMALIZATION ###############################################################

class BatchNormalization(tf.keras.layers.Layer):
    def __init__(
        self,
        axis: int=0,
        momentum: float=0.99,
        epsilon: float=0.001,
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

    def build(self, input_shape: tuple):
        # shape
        __axis = self._axis % len(input_shape) # positive index even when the axis is specified negatively, like -2
        __shape = [1 if __i == __axis else __d for __i, __d in enumerate(input_shape)]
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
        # notify the model
        self.built = True

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
        axis: int=-1,
        epsilon: float=0.001,
        **kwargs
    ):
        super(LayerNormalization, self).__init__(**kwargs)
        self._axis = axis
        self._epsilon = epsilon
        self._gain = None
        self._bias = None

    def build(self, input_shape: tuple):
        # shape
        __shape = [1] + input_shape[1:]
        # values
        __gain_init = _mti.SmallNormal()
        __bias_init = _mti.SmallNormal()
        # tensors
        self._gain = self.add_weight("gain", shape=__shape, initializer=__gain_init)
        self._bias = self.add_weight("bias", shape=__shape, initializer=__bias_init)
        # notify the model
        self.built = True

    def call(self, inputs: tf.Tensor, training: bool=True, **kwargs):
        # current values
        __layer_mean = tf.math.reduce_mean(inputs, axis=self._axis, keepdims=True)
        __layer_stddev = tf.math.reduce_std(inputs, axis=self._axis, keepdims=True)
        # normalize
        __normalized = tf.math.divide(inputs - __layer_mean, __layer_stddev + self._epsilon)
        # scale
        return tf.math.add(tf.math.multiply(self._gain, __normalized), self._bias)

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

    def build(self, input_shape: tuple):
        # kernel
        __kernel_init = _mti.SmallNormal()
        self._kernel = self.add_weight("kernel", shape=[int(input_shape[-1]), self._units], initializer=__kernel_init)
        # bias
        if self._biased:
            __bias_init = _mti.SmallNormal()
            self._bias = self.add_weight("bias", shape=[self._units], initializer=__bias_init)
        # notify the model
        self.built = True

    def call(self, inputs: tf.Tensor, **kwargs):
        return tf.matmul(inputs, self._kernel) + self._bias if (self._biased and self._bias is not None) else tf.matmul(inputs, self._kernel)

# QUADRATIC ###################################################################

class Attention(tf.keras.layers.Layer):
    def __init__(
        self,
        head_dim: int,
        head_count: int=1,
        **kwargs
    ):
        super(Attention, self).__init__(**kwargs)
        self._time_dim = None
        self._head_dim = head_dim
        self._head_count = head_count
        self._key = None
        self._query = None
        self._value = None

    def build(self, input_shape: tuple) -> None:
        self._time_dim = list(input_shape)[-2]
        # init
        __key_init = _mti.SmallNormal()
        __query_init = _mti.SmallNormal()
        __value_init = _mti.SmallNormal()
        # kernels
        self._key = self.add_weight("key", shape=[int(input_shape[-1]), self._head_dim], initializer=__key_init)
        self._query = self.add_weight("query", shape=[int(input_shape[-1]), self._head_dim], initializer=__query_init)
        self._value = self.add_weight("value", shape=[self._time_dim, self._head_dim], initializer=__value_init)
        # notify the model
        self.built = True

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        # transpose the last two axes
        __perm = list(range(len(list(inputs.shape))))
        __perm[-1] = len(__perm) - 2
        __perm[-2] = len(__perm) - 1
        # key
        __k = tf.matmul(inputs, self._key) # (B, T, E) * (E, H) = (B, T, H)
        # query
        __q = tf.matmul(inputs, self._query) # (B, T, E) * (E, H) = (B, T, H)
        # weight
        __w = tf.matmul(__k, tf.transpose(__q, perm=__perm)) / tf.math.sqrt(float(self._head_dim)) # (B, T, H) * (B, H, T) = (B, T, T)
        # mask
        __m = tf.linalg.band_part(tf.ones((self._time_dim, self._time_dim)), num_lower=0, num_upper=-1) - tf.linalg.diag(self._time_dim * [1.]) # (T, T)
        __u = tf.where(__m == 1., -np.inf, 0.) # (T, T)
        __l = tf.linalg.band_part(__w, num_lower=-1, num_upper=0) # (B, T, T) may fail because of the first dimension => diag of tensor with 3 axes
        # probabilities
        __w = tf.nn.softmax(__u + __l, axis=-1) # (T, T) + (B, T, T) = (B, T, T)
        # value
        return tf.matmul(__w, self._value) # (B, T, T) * (T, H) = (B, T, H)

# EMBEDDING ###################################################################

class Embedding(tf.keras.layers.Layer):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        **kwargs
    ):
        super(Embedding, self).__init__(**kwargs)
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._kernel = None
        self._position_kernel = None

    def build(self, input_shape: tuple):
        __kernel_init = _mti.SmallNormal()
        # register the weights
        self._kernel = self.add_weight(name="kernel", shape=[self._input_dim, self._output_dim], initializer=__kernel_init)
        # notify the model
        self.built = True

    def call(self, inputs: tf.Tensor, **kwargs):
        # content embedding
        __x = tf.one_hot(indices=inputs, depth=self._input_dim, dtype=tf.dtypes.float32)
        return tf.matmul(__x, self._kernel)

class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(
        self,
        input_axis: int=1, # axis of the sequence
        output_axis: int=-1, # axis of the embedding
        **kwargs
    ):
        super(PositionalEmbedding, self).__init__(**kwargs)
        self._input_axis = input_axis
        self._output_axis = output_axis
        self._kernel = None

    def build(self, input_shape: tuple):
        # shape
        __axes = [self._input_axis % len(input_shape), self._output_axis % len(input_shape)]
        __shape = [(__d if __i in __axes else 1) for __i, __d in enumerate(list(input_shape))]
        # init values
        __kernel_init = _mti.SmallNormal()
        # register the weights
        self._kernel = self.add_weight(name="kernel", shape=__shape, initializer=__kernel_init)
        # notify the model
        self.built = True

    def call(self, inputs: tf.Tensor):
        return inputs + self._kernel # each index in the sequence axis has a dedicated bias (different from dense bias)

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

def _normalize_shape(shape: list) -> list:
    return [-1 if __d is None else __d for __d in shape]

def _normalize_dim(dim: int) -> int:
    return -1 if (dim is None or dim < 0) else dim

def _multiply_dim(dim_l: int, dim_r: int) -> int:
    return -1 if (dim_l == -1 or dim_r == -1) else dim_l * dim_r

def _divide_dim(dim_l: int, dim_r: int) -> int:
    return -1 if (dim_l == -1 or dim_r == -1) else dim_l // dim_r

class Divide(tf.keras.layers.Layer):
    def __init__(
        self,
        input_axis: int, # relative to the NEW shape / rank
        output_axis: int, # same
        factor: int,
        insert: bool=False,
        **kwargs
    ) -> None:
        super(Divide, self).__init__(**kwargs)
        self._input_axis = input_axis
        self._output_axis = output_axis
        self._factor = factor
        self._insert = insert

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        # infer the dimension of the symbolic axis
        __shape = _normalize_shape(list(inputs.shape))
        # rank, according to the new shape
        __rank = len(__shape) + int(self._insert)
        # axes, taken from the new shape
        __axis0 = self._input_axis % __rank
        __axis1 = self._output_axis % __rank
        # option to group data on a new axis
        if self._insert: __shape.insert(__axis1, 1)
        # move data from axis 0 to axis 1
        __shape[__axis0] = _divide_dim(__shape[__axis0], self._factor)
        __shape[__axis1] = _multiply_dim(__shape[__axis1], self._factor)
        return tf.reshape(tensor=inputs, shape=__shape)

class Merge(tf.keras.layers.Layer):
    def __init__(
        self,
        left_axis: int=-2,
        right_axis: int=-1,
        left: bool=True,
        **kwargs
    ) -> None:
        super(Merge, self).__init__(**kwargs)
        self._left_axis = left_axis
        self._right_axis = right_axis
        self._left = left

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        # infer the dimension of the symbolic axis
        __shape = _normalize_shape(list(inputs.shape))
        __rank = len(__shape)
        # target axes
        __axis_l = self._left_axis % __rank
        __axis_r = self._right_axis % __rank
        # new axis
        __dim = _multiply_dim(__shape[__axis_l], __shape[__axis_r])
        __axis_k = __axis_l if self._left else __axis_r # kept axis
        __axis_d = __axis_r if self._left else __axis_l # deleted axis
        # new shape
        __shape[__axis_k] = __dim
        __shape.pop(__axis_d)
        # actually merge the two axes
        return tf.reshape(tensor=inputs, shape=__shape)

class Reshape(tf.keras.layers.Layer):
    def __init__(
        self,
        target_shape: tuple,
        **kwargs
    ) -> None:
        super(Reshape, self).__init__(**kwargs)
        self._shape = target_shape

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        return tf.reshape(inputs, self._shape)
