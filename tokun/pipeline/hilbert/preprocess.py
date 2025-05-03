import functools

import tensorflow as tf

import mlable.maths.ops
import mlable.shaping.axes
import mlable.shaping.hilbert
import mlable.text

# CONSTANTS ####################################################################

ANSI_REGEX = r'\x1b\[[0-9;]*[mGKHF]'

# PREPROCESS ###################################################################

def _parser_factory(features: list=[], separator: str='\x1d') -> callable:
    # select the relevant features
    __list = lambda __sample: [__sample[__f] for __f in features]
    # join them
    __join = functools.partial(tf.strings.join, separator=separator)
    # ignore if no features were given
    __list = __list if features else lambda __x: __x
    __join = __join if features else lambda __x: __x
    # inputs = targets for the autoencoders
    def __parser(inputs) -> tuple:
        return (__join(__list(inputs)), __join(__list(inputs)))
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

def _encoder_factory(sample_dim: int, encoding: str='UTF-8') -> callable:
    # text encoding (UTF-32-BE or UTF-8)
    __utf = functools.partial(mlable.text.encode, sample_dim=sample_dim, output_dtype=tf.uint8, output_encoding=encoding)
    # encode all
    def __encoder(inputs: tf.Tensor, targets: tf.Tensor) -> tuple:
        return (__utf(inputs), __utf(targets))
    # customized fn
    return __encoder

def _formatter_factory(batch_dim: int, token_dim: int, order_num: int, rank_num: int) -> callable:
    # entire shape after trimming
    __shape = (batch_dim,) + rank_num * (1 << order_num,) + (token_dim,)
    # group the bytes token by token
    __group = functools.partial(mlable.shaping.axes.divide, axis=1, factor=token_dim, insert=True, right=True)
    # fold along the Hilbert curve
    __fold = functools.partial(mlable.shaping.hilbert.fold, order=order_num, rank=rank_num, axis=1)
    # enforce types
    __cast_i = functools.partial(tf.cast, dtype=tf.uint8)
    __cast_t = functools.partial(tf.cast, dtype=tf.float32)
    # enforce shapes
    __reshape = functools.partial(tf.reshape, shape=__shape)
    # chain the operations
    def __formatter(inputs: tf.Tensor, targets: tf.Tensor) -> tuple:
        return (__cast_i(__reshape(__fold(__group(inputs)))), __cast_t(__reshape(__fold(__group(targets)))))
    # customized fn
    return __formatter

def _embedder_factory(bigendian: bool=True) -> callable:
    # embed all
    def __embedder(inputs: tf.Tensor, targets: tf.Tensor) -> tuple:
        return (inputs, mlable.shaping.axes.merge(mlable.maths.ops.expand_base(targets, base=2, depth=8, bigendian=bigendian), axis=-1, right=False))
    # customized fn
    return __embedder

# > END-TO-END #################################################################

def _wrapper(inputs: tf.Tensor, parser: callable, cleaner: callable, encoder: callable, embedder: callable, formatter: callable) -> tuple:
    # fetch the relevant features
    __inputs, __targets = parser(inputs=inputs)
    # sanitize
    __inputs, __targets = cleaner(inputs=__inputs, targets=__targets)
    # encode / tokenize
    __inputs, __targets = encoder(inputs=__inputs, targets=__targets)
    # enforce types + shapes
    __inputs, __targets = formatter(inputs=__inputs, targets=__targets)
    # represent the output in binary
    __inputs, __targets = embedder(inputs=__inputs, targets=__targets)
    # targets = inputs (in binary) for the autoencoder
    return (__inputs, __targets) # __weights

def factory(batch_dim: int, token_dim: int, order_num: int, rank_num: int, features: list=[], pattern: str=ANSI_REGEX, rewrite: str='', separator: str='\x1d', encoding: str='UTF-8', bigendian: bool=True) -> callable:
    __sample_dim = token_dim * (1 << (rank_num * order_num))
    # custom fn
    __parser = _parser_factory(features=features, separator=separator)
    __cleaner = _cleaner_factory(pattern=pattern, rewrite=rewrite)
    __encoder = _encoder_factory(sample_dim=__sample_dim, encoding=encoding)
    __formatter = _formatter_factory(batch_dim=batch_dim, token_dim=token_dim, order_num=order_num, rank_num=rank_num)
    __embedder = _embedder_factory(bigendian=bigendian)
    # actual preprocessing function
    return functools.partial(_wrapper, parser=__parser, cleaner=__cleaner, encoder=__encoder, embedder=__embedder, formatter=__formatter)
