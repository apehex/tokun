import functools

import tensorflow as tf

import mlable.ops
import mlable.shaping
import tokun.pipeline.text

# MASK #########################################################################

def mask(data: tf.Tensor, padding_value: int=0, padding_weight: float=0.0, data_weight: float=1.0, dtype: tf.dtypes.DType=tf.float32) -> tf.Tensor:
    # byte level mask
    __weights = tf.not_equal(data, padding_value)
    # instruction level mask, but expressed byte by byte
    __weights = mlable.ops.reduce_any(data=__weights, group=None, axis=-1, keepdims=False)
    # cast from bool to allow multiplications
    __weights = tf.cast(__weights, dtype=dtype)
    # rescale the weights
    return data_weight * __weights + padding_weight * (1. - __weights)

# PREPROCESS ###################################################################

def _parser_factory(token_dim: int, features: list, separator: str='\x1d') -> callable:
    def __parser(inputs) -> tuple:
        # fetch the relevant features
        __inputs = tf.strings.join(inputs=[inputs[__f] for __f in features], separator=separator)
        # (input, target) objective = reconstructing the input
        return (__inputs, __inputs) # tokun.pipeline.text.offset(data=__inputs, ticks=token_dim // 4) # the target is the next token for each input
    # customized fn
    return __parser

def _encoder_factory(token_dim: int, sample_dim: int) -> callable:
    # text encoding (UTF-32-BE)
    __utf32 = functools.partial(tokun.pipeline.text.encode, token_dim=token_dim, sample_dim=sample_dim, output_dtype=tf.uint8)
    # encode all
    def __encoder(inputs: tf.Tensor, targets: tf.Tensor) -> tuple:
        return (__utf32(inputs), __utf32(targets))
    # customized fn
    return __encoder

def _formatter_factory(batch_dim: int, sample_dim: int) -> callable:
    # enforce types
    __cast_i = functools.partial(tf.cast, dtype=tf.int32)
    __cast_t = functools.partial(tf.cast, dtype=tf.float32)
    # enforce shapes
    __reshape = functools.partial(tf.reshape, shape=(batch_dim, sample_dim))
    # chain the operations
    def __formatter(inputs: tf.Tensor, targets: tf.Tensor) -> tuple:
        return (__cast_i(__reshape(inputs)), __cast_t(__reshape(targets)))
    # customized fn
    return __formatter

def _embedder_factory() -> callable:
    # embed all
    def __embedder(inputs: tf.Tensor, targets: tf.Tensor) -> tuple:
        return (inputs, mlable.ops.expand_base(targets, base=2, depth=8))
    # customized fn
    return __embedder

def _masker_factory(data_weight: float=1.0, padding_weight: float=0.0) -> callable:
    def __masker(inputs: tf.Tensor) -> tf.Tensor:
        return mask(data=inputs, padding_value=0, data_weight=data_weight, padding_weight=padding_weight, dtype=tf.float32)
    # customized fn
    return __masker

# > END-TO-END #################################################################

def _wrapper(inputs: tf.Tensor, parser: callable, encoder: callable, embedder: callable, formatter: callable) -> tuple: # masker: callable
    # fetch the relevant features
    __inputs, __targets = parser(inputs=inputs)
    # encode / tokenize
    __inputs, __targets = encoder(inputs=__inputs, targets=__targets)
    # enforce types + shapes
    __inputs, __targets = formatter(inputs=__inputs, targets=__targets)
    # embed with tokun
    __inputs, __targets = embedder(inputs=__inputs, targets=__targets)
    # sequence mask to ignore padding during training
    # __weights = masker(inputs=__inputs)
    # pack both sourcecode and bytecode into the model inputs
    return (__inputs, __targets) # __weights

def factory(batch_dim: int, sample_dim: int, token_dim: int, features: list, separator: str='\x1d') -> callable: # data_weight: float=1.0, padding_weight: float=0.0
    # custom fn
    __parser = _parser_factory(token_dim=token_dim, features=features, separator=separator)
    __encoder = _encoder_factory(sample_dim=sample_dim, token_dim=token_dim)
    __formatter = _formatter_factory(batch_dim=batch_dim, sample_dim=sample_dim)
    __embedder = _embedder_factory()
    # __masker = _masker_factory(data_weight=data_weight, padding_weight=padding_weight)
    # actual preprocessing function
    return functools.partial(_wrapper, parser=__parser, encoder=__encoder, embedder=__embedder, formatter=__formatter) # masker=__masker
