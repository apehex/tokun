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

def encode(data: tf.Tensor, token_dim: int, sample_dim: int, output_dtype: tf.DType=tf.uint8, output_encoding: str='UTF-32-BE') -> tf.Tensor:
    # factor 4 because of the UTF-32 encoding
    __dim = math.ceil(sample_dim / token_dim) * token_dim
    # decode bytes from UTF-8
    __bytes = tf.strings.unicode_transcode(input=data, input_encoding='UTF-8', output_encoding=output_encoding) # (B,)
    # decode byte strings to arrays of byte integers
    return tf.io.decode_raw(__bytes, out_type=output_dtype, fixed_length=__dim, little_endian=False) # (B, 4 * S) or (B, S) depending on the dtype (1 or 4 bytes)

# DROP #########################################################################

def trim(data: tf.Tensor, count: int=1) -> tf.Tensor:
    # group the bytes 4 by 4 (one UTF-32 character)
    __outputs = mlable.shaping.divide(data, input_axis=-2, output_axis=-1, factor=4, insert=True)
    # remove the most significant bytes (most often 0 in UTF-32)
    __outputs = tf.gather(__outputs, indices=range(count, 4), axis=-1)
    # flatten the data back
    return mlable.shaping.merge(__outputs, left_axis=-2, right_axis=-1, left=True)

def untrim(data: tf.Tensor, count: int=1) -> tf.Tensor:
    # group the bytes codepoint by codepoint (4 bytes minus the ones that were trimmed)
    __outputs = mlable.shaping.divide(data, input_axis=-2, output_axis=-1, factor=4 - count, insert=True)
    # add leading 0s to each group / codepoint
    __outputs = tf.concat([tf.zeros_like(__outputs[..., :count], dtype=__outputs.dtype), __outputs[..., :]], axis=-1)
    # flatten the data back
    return mlable.shaping.merge(__outputs, left_axis=-2, right_axis=-1, left=True)

# AUGMENT ######################################################################

def offset(data: tf.Tensor, ticks: int=1) -> tf.Tensor:
    return tf.convert_to_tensor([ticks * b'\x00']) + data

# DECODE #######################################################################

def codepoint(data: tf.Tensor) -> tf.Tensor:
    # make sure the dtype is large enough for UTF-32 codepoints
    __data = tf.cast(data, dtype=tf.int32)
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

def preprocess(text: str, token_dim: int, expand_dims: list=[1], encode_dtype: tf.DType=tf.uint8, output_dtype: tf.DType=tf.uint8, output_encoding: str='UTF-32-BE') -> tf.Tensor:
    # as tensor
    __data = tf.convert_to_tensor(text, dtype=tf.string)
    # list of bytes / codepoints
    __bytes = encode(data=__data, token_dim=token_dim, sample_dim=4 * len(text), output_dtype=encode_dtype, output_encoding=output_encoding)
    # expand with unitary batch dim + cast
    return tf.cast(tf.expand_dims(__bytes, axis=0), dtype=output_dtype)

# < ############################################################################

def unpad(text: str) -> str:
    return text.strip('\x00')

def unpack(data: tf.Tensor) -> list:
    __data = data.numpy().tolist()
    return [__s.decode('utf-8') for __s in __data]

def postprocess(logits: tf.Tensor, threshold: float=0.0, temp: float=1.0, topp: float=0.0, topk: int=0, dtype: tf.DType=tf.uint8) -> tf.Tensor:
    __outputs = mlable.sampling.binary(logits=logits, threshold=threshold, temp=temp, topp=topp, topk=topk, dtype=dtype)
    # merge the bytes into codepoints
    __outputs = codepoint(data=__outputs)
    # decode the UTF-32-BE codepoints
    return decode(data=__outputs)

# SAMPLING #####################################################################

def sample(model: tf.keras.models.Model, text: str, **kwargs) -> tuple:
    __x = preprocess(text=text, token_dim=kwargs.get('token_dim', 16), expand_dims=kwargs.get('expand_dims', [1]), output_dtype=kwargs.get('output_dtype', tf.uint8))
    __e = model.encode(__x)
    __p = model.decode(__e)
    __y = postprocess(__p, threshold=kwargs.get('threshold', 0.5), random=kwargs.get('random', False))
    __o = unpack(__y)
    return (__x, __e, __p, __y, __o)
