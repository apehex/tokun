import functools

import tensorflow as tf

import mlable.maths.ops
import mlable.shaping.axes
import mlable.text

# CONSTANTS ####################################################################

ANSI_REGEX = r'\x1b\[[0-9;]*[mGKHF]'

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

def _encoder_factory(sample_dim: int, encoding: str='UTF-32-BE') -> callable:
    # text encoding (UTF-32-BE or UTF-8)
    __utf = functools.partial(mlable.text.encode, sample_dim=sample_dim, output_dtype=tf.uint8, output_encoding=encoding)
    # encode all
    def __encoder(inputs: tf.Tensor, targets: tf.Tensor) -> tuple:
        return (__utf(inputs), __utf(targets))
    # customized fn
    return __encoder

def _formatter_factory(batch_dim: int, sample_dim: int, token_dim: int=1, drop_dim: int=0, encoding_dim: int=4) -> callable:
    # sample dimension after trimming
    __sample_dim = max(1, encoding_dim - drop_dim) * (sample_dim // encoding_dim)
    # sample dimension after grouping by token
    __sample_dim = __sample_dim // max(1, token_dim)
    # final shape
    __shape = (batch_dim, __sample_dim, token_dim) if (token_dim > 1) else (batch_dim, __sample_dim)
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

def _embedder_factory(bigendian: bool=True) -> callable:
    # embed all
    def __embedder(inputs: tf.Tensor, targets: tf.Tensor) -> tuple:
        return (inputs, mlable.shaping.axes.merge(mlable.maths.ops.expand_base(targets, base=2, depth=8, bigendian=bigendian), axis=-1, right=False))
    # customized fn
    return __embedder

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
    # encode the targets in binary
    __inputs, __targets = embedder(inputs=__inputs, targets=__targets)
    # targets = inputs (in binary) for the autoencoder
    return (__inputs, __targets)

def factory(batch_dim: int, sample_dim: int, features: list, pattern: str=ANSI_REGEX, rewrite: str='', separator: str='\x1d', encoding: str='UTF-32-BE', bigendian: bool=True, token_dim: int=1, drop_dim: int=0) -> callable:
    __encoding_dim = 4 if '32' in encoding else 1
    # custom fn
    __parser = _parser_factory(features=features, separator=separator)
    __cleaner = _cleaner_factory(pattern=pattern, rewrite=rewrite)
    __encoder = _encoder_factory(sample_dim=sample_dim, encoding=encoding)
    __formatter = _formatter_factory(batch_dim=batch_dim, sample_dim=sample_dim, token_dim=token_dim, drop_dim=drop_dim, encoding_dim=__encoding_dim)
    __embedder = _embedder_factory(bigendian=bigendian)
    # actual preprocessing function
    return functools.partial(_wrapper, parser=__parser, cleaner=__cleaner, encoder=__encoder, embedder=__embedder, formatter=__formatter)
