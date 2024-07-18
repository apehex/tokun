"""Pre and post processing pipelines for tokun."""

import functools
import itertools
import math

import tensorflow as tf

import mlable.ops
import mlable.sampling
import mlable.utils

# UNICODE #####################################################################

CODE_STX = b'\x02'
CODE_ETX = b'\x03'
CODE_FS = b'\x1c'
CODE_GS = b'\x1d'
CODE_RS = b'\x1e'
CODE_US = b'\x1f'

# ENCODE ######################################################################

def encode(data: tf.Tensor, token_size: int, sample_size: int, dtype: tf.dtypes.DType=tf.dtypes.int32) -> tf.Tensor:
    # factor 4 because of the UTF-32 encoding
    __dim = math.ceil(4 * sample_size / token_size) * token_size
    # decode bytes from UTF-8
    __bytes = tf.strings.unicode_transcode(input=data, input_encoding='UTF-8', output_encoding='UTF-32-BE') # (B,)
    # decode byte strings to arrays of byte integers
    __bytes = tf.io.decode_raw(__bytes, out_type=tf.uint8, fixed_length=__dim) # (B, 4 * S)
    # cast to int32 as uint8 is not std
    return tf.cast(__bytes, dtype=dtype)

# RESHAPE #####################################################################

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

# AUGMENT #####################################################################

def offset(data: tf.Tensor, ticks: int=1) -> tf.Tensor:
    return tf.convert_to_tensor([ticks * b'\x00']) + data

# DECODE ######################################################################

def decode(data: tf.Tensor) -> str:
    # make sure the dtype is large enough for UTF-32 codepoints
    __data = tf.cast(data, dtype=tf.dtypes.int32)
    # group the bytes 4 by 4
    __shape = mlable.utils.divide_shape(shape=__data.shape, input_axis=-2, output_axis=-1, factor=4, insert=True)
    __bytes = tf.reshape(tensor=__data, shape=__shape)
    # compute the UTF-32-BE codepoints
    __codes = mlable.ops.reduce_base(data=__bytes, base=256, axis=-1, keepdims=False)
    # actually decode
    __utf32 = tf.strings.unicode_encode(__codes, output_encoding='UTF-32-BE')
    # convert to standard UTF-8
    return tf.strings.unicode_transcode(input=__utf32, input_encoding='UTF-32-BE', output_encoding='UTF-8')

# > ###########################################################################

def preprocess(text: str, token_size: int, expand: list=[]) -> tf.Tensor:
    # as tensor
    __data = tf.convert_to_tensor(text, dtype=tf.dtypes.string)
    # list of bytes
    __bytes = encode(data=__data, token_size=token_size, sample_size=len(text))
    # partition or flatten
    return reshape(data=__bytes, expand=expand)

# < ###########################################################################

def unpad(text: str) -> str:
    return text.strip('\x00')

def unpack(data: tf.Tensor) -> list:
    __data = data.numpy().tolist()
    return [__s.decode('utf-8') for __s in __data]

def postprocess(prediction: tf.Tensor, binary: bool=False, random: bool=False) -> str:
    # from one-hot to UTF-32 bytes
    __output = mlable.sampling.binary(prediction=prediction, threshold=0.5, random=random) if binary else mlable.sampling.categorical(prediction=prediction, random=random)
    # flatten the groups of 4 bytes
    return decode(data=__output)

# SAMPLING ####################################################################

def sample(model: tf.keras.models.Model, text: str, **kwargs) -> tuple:
    __x = preprocess(text=text, token_size=kwargs.get('token_size', 16), expand=kwargs.get('expand', [1]))
    __e = model._encoder(__x)
    __p = model._decoder(__e)
    __y = postprocess(__p, binary=kwargs.get('binary', False), random=kwargs.get('random', False))
    __o = unpack(__y)
    return (__x, __e, __p, __y, __o)
