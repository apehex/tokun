"""Pre and post processing pipelines for tokun."""

import functools
import itertools
import math

import tensorflow as tf

import mlable.ops
import mlable.sampling
import mlable.shaping

# UNICODE ######################################################################

CODE_STX = b'\x02'
CODE_ETX = b'\x03'
CODE_FS = b'\x1c'
CODE_GS = b'\x1d'
CODE_RS = b'\x1e'
CODE_US = b'\x1f'

# SPLIT ########################################################################

def split(data: tf.Tensor, height_dim: int, separator_str: str='\n', padding_str: str='') -> tf.Tensor:
    # add an axis for the substrings
    __shape = tuple(data.shape) + (height_dim,)
    # don't limit the number of splits yet
    __outputs = tf.strings.split(data, sep=separator_str, maxsplit=-1)
    # pad and truncate to enforce the shape
    return __outputs.to_tensor(default_value=padding_str, shape=__shape)

# ENCODE #######################################################################

def encode(data: tf.Tensor, token_dim: int, sample_dim: int, output_dtype: tf.dtypes.DType=tf.uint8) -> tf.Tensor:
    # factor 4 because of the UTF-32 encoding
    __dim = math.ceil(sample_dim / token_dim) * token_dim
    # decode bytes from UTF-8
    __bytes = tf.strings.unicode_transcode(input=data, input_encoding='UTF-8', output_encoding='UTF-32-BE') # (B,)
    # decode byte strings to arrays of byte integers
    return tf.io.decode_raw(__bytes, out_type=output_dtype, fixed_length=__dim, little_endian=False) # (B, 4 * S) or (B, S) depending on the dtype (1 or 4 bytes)

# RESHAPE ######################################################################

def chunk(seq: list, size: int, repeats: bool=True) -> list:
    __chunks = (seq[__i:__i+size] for __i in range(0, len(seq), size))
    return list(__chunks if repeats else set(__chunks))

def merge(chunks: list) -> list:
    return list(itertools.chain.from_iterable(chunks))

def shape(expand: list=[]) -> list:
    return expand + [-1]

def reshape(data: tf.Tensor, expand: list=[]) -> tf.Tensor:
    # group by token unit
    __shape = shape(expand=expand)
    # partition or flatten the data
    return tf.reshape(tensor=data, shape=__shape) # for example (-1, G, G, G) the first dimension is not B

# AUGMENT ######################################################################

def offset(data: tf.Tensor, ticks: int=1) -> tf.Tensor:
    return tf.convert_to_tensor([ticks * b'\x00']) + data

# DECODE #######################################################################

def codepoint(data: tf.Tensor) -> tf.Tensor:
    # make sure the dtype is large enough for UTF-32 codepoints
    __data = tf.cast(data, dtype=tf.dtypes.int32)
    # group the bytes 4 by 4
    __bytes = mlable.shaping.divide(data=__data, input_axis=-2, output_axis=-1, factor=4, insert=True)
    # compute the UTF-32-BE codepoints
    return mlable.ops.reduce_base(data=__bytes, base=256, axis=-1, keepdims=False)

def decode(data: tf.Tensor) -> tf.Tensor:
    # input = array of unicode codepoints
    __utf32 = tf.strings.unicode_encode(data, output_encoding='UTF-32-BE')
    # convert to standard UTF-8
    return tf.strings.unicode_transcode(input=__utf32, input_encoding='UTF-32-BE', output_encoding='UTF-8')

# > ############################################################################

def preprocess(text: str, token_dim: int, expand_dims: list=[1], encode_dtype: tf.dtypes.DType=tf.uint8, output_dtype: tf.dtypes.DType=tf.int32) -> tf.Tensor:
    # as tensor
    __data = tf.convert_to_tensor(text, dtype=tf.dtypes.string)
    # list of bytes / codepoints
    __bytes = encode(data=__data, token_dim=token_dim, sample_dim=4 * len(text), output_dtype=encode_dtype)
    # expand with unitary batch dim + cast
    return tf.cast(reshape(data=__bytes, expand=expand_dims), dtype=output_dtype)

# < ############################################################################

def unpad(text: str) -> str:
    return text.strip('\x00')

def unpack(data: tf.Tensor) -> list:
    __data = data.numpy().tolist()
    return [__s.decode('utf-8') for __s in __data]

def postprocess(prediction: tf.Tensor, threshold: float=0.5, random: bool=False) -> tf.Tensor:
    __output = mlable.sampling.binary(prediction=prediction, threshold=threshold, random=random)
    # merge the bytes into codepoints
    __output = codepoint(data=__output)
    # decode the UTF-32-BE codepoints
    return decode(data=__output)

# SAMPLING #####################################################################

def sample(model: tf.keras.models.Model, text: str, **kwargs) -> tuple:
    __x = preprocess(text=text, token_dim=kwargs.get('token_dim', 16), expand_dims=kwargs.get('expand_dims', [1]), output_dtype=kwargs.get('output_dtype', tf.uint8))
    __e = model.encode(__x)
    __p = model.decode(__e)
    __y = postprocess(__p, threshold=kwargs.get('threshold', 0.5), random=kwargs.get('random', False))
    __o = unpack(__y)
    return (__x, __e, __p, __y, __o)
