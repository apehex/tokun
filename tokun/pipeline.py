import functools

import tensorflow as tf

import mlable.maths.ops
import mlable.text

# CONSTANTS ####################################################################

ANSI_REGEX = r'\x1b\[[0-9;]*[mGKHF]'

# MASK #########################################################################

def mask(data: tf.Tensor, encoding_dim: int, padding_value: int=0, padding_weight: float=0.0, data_weight: float=1.0, dtype: tf.DType=tf.float32) -> tf.Tensor:
    # byte level mask (B, S * T)
    __weights = tf.not_equal(data, padding_value)
    # token level mask, but expressed byte by byte
    __weights = mlable.maths.ops.reduce_any(data=__weights, group=encoding_dim, axis=-1, keepdims=True)
    # cast from bool to allow multiplications
    __weights = tf.cast(__weights, dtype=dtype)
    # rescale the weights
    return data_weight * __weights + padding_weight * (1. - __weights)

# PREPROCESS ###################################################################

def _parser_factory(features: list, separator: str='\x1d') -> callable:
    def __parser(inputs) -> tuple:
        # fetch the relevant features
        __inputs = tf.strings.join(inputs=[inputs[__f] for __f in features], separator=separator)
        # (input, target) objective = reconstructing the input
        return (__inputs, __inputs)
    # customized fn
    return __parser

def _cleaner_factory(pattern: str=ANSI_REGEX, rewrite: str='') -> callable:
    __replace = functools.partial(tf.strings.regex_replace, pattern=pattern, rewrite=rewrite, replace_global=True)
    # do nothing when the pattern is empty
    __replace = __replace if pattern else (lambda __x: __x)
    # chain the operations
    def __cleaner(inputs: tf.Tensor, targets: tf.Tensor) -> tuple:
        return (__replace(inputs), __replace(targets))
    # customized fn
    return __cleaner

def _encoder_factory(height_dim: int, width_dim: int, encoding: str='UTF-32-BE') -> callable:
    # split each sample into substrings
    __split = functools.partial(mlable.text.split, height_dim=height_dim, separator_str='\n', padding_str='')
    # ignore when the output is flat
    __split = __split if (height_dim > 1) else (lambda __x: __x)
    # text encoding (UTF-32-BE or UTF-8)
    __utf = functools.partial(mlable.text.encode, sample_dim=width_dim, output_dtype=tf.uint8, output_encoding=encoding)
    # encode all
    def __encoder(inputs: tf.Tensor, targets: tf.Tensor) -> tuple:
        return (__utf(__split(inputs)), __utf(__split(targets)))
    # customized fn
    return __encoder

def _formatter_factory(batch_dim: int, height_dim: int, width_dim: int, drop_dim: int=0, encoding_dim: int=4) -> callable:
    # sample dimension after trimming
    __width_dim = max(1, encoding_dim - drop_dim) * (width_dim // encoding_dim)
    # entire shape after trimming
    __shape = (batch_dim, height_dim, __width_dim) if (height_dim > 0) else (batch_dim, __width_dim)
    # remove the leading 0s in UTF-32-BE
    __trim = functools.partial(mlable.text.trim, count=drop_dim, outof=encoding_dim)
    # enforce types
    __cast_i = functools.partial(tf.cast, dtype=tf.uint8)
    __cast_t = functools.partial(tf.cast, dtype=tf.float32)
    # enforce shapes
    __reshape = functools.partial(tf.reshape, shape=__shape)
    # chain the operations
    def __formatter(inputs: tf.Tensor, targets: tf.Tensor) -> tuple:
        return (__cast_i(__reshape(__trim(inputs))), __cast_t(__reshape(__trim(targets))))
    # customized fn
    return __formatter

def _embedder_factory() -> callable:
    # embed all
    def __embedder(inputs: tf.Tensor, targets: tf.Tensor) -> tuple:
        return (inputs, mlable.maths.ops.expand_base(targets, base=2, depth=8))
    # customized fn
    return __embedder

def _masker_factory(encoding_dim: int, data_weight: float=1.0, padding_weight: float=0.0) -> callable:
    def __masker(inputs: tf.Tensor) -> tf.Tensor:
        return mask(data=inputs, encoding_dim=encoding_dim, padding_value=0, data_weight=data_weight, padding_weight=padding_weight, dtype=tf.float32)
    # customized fn
    return __masker

# > END-TO-END #################################################################

def _wrapper(inputs: tf.Tensor, parser: callable, cleaner: callable, encoder: callable, embedder: callable, formatter: callable) -> tuple: # masker: callable
    # fetch the relevant features
    __inputs, __targets = parser(inputs=inputs)
    # sanitize
    __inputs, __targets = cleaner(inputs=__inputs, targets=__targets)
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

def factory(batch_dim: int, height_dim: int, width_dim: int, features: list, pattern: str=ANSI_REGEX, rewrite: str='', separator: str='\x1d', encoding: str='UTF-32-BE', drop_dim: int=0) -> callable: # data_weight: float=1.0, padding_weight: float=0.0
    __encoding_dim = 4 if '32' in encoding else 1
    # custom fn
    __parser = _parser_factory(features=features, separator=separator)
    __cleaner = _cleaner_factory(pattern=pattern, rewrite=rewrite)
    __encoder = _encoder_factory(height_dim=height_dim, width_dim=width_dim, encoding=encoding)
    __formatter = _formatter_factory(batch_dim=batch_dim, height_dim=height_dim, width_dim=width_dim, drop_dim=drop_dim, encoding_dim=__encoding_dim)
    __embedder = _embedder_factory()
    # __masker = _masker_factory(encoding_dim=__encoding_dim, data_weight=data_weight, padding_weight=padding_weight)
    # actual preprocessing function
    return functools.partial(_wrapper, parser=__parser, cleaner=__cleaner, encoder=__encoder, embedder=__embedder, formatter=__formatter) # masker=__masker
