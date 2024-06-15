"""Pre and post processing pipelines for tokun."""

import functools
import itertools
import math

import tensorflow as tf

# ENCODE ######################################################################

def _encode_scalar(text: str, token_size: int) -> tf.Tensor:
    # encode the string
    __bytes = list(text.encode('utf-32-be'))
    # pad until the encodeed text has length multiple of the token length
    __padding = (-len(__bytes) % token_size) * [0]
    # concat data and padding
    return tf.convert_to_tensor(value=__bytes + __padding, dtype=tf.dtypes.int32) # uint8 is not allowed

def _encode_tensor(data: tf.Tensor, token_size: int, sample_size: int=64) -> tf.Tensor:
    # factor 4 because of the UTF-32 encoding
    __dim = math.ceil(4 * sample_size / token_size) * token_size
    # Decode bytes from UTF-8
    __bytes = tf.strings.unicode_transcode(input=data, input_encoding='UTF-8', output_encoding='UTF-32-BE') # (B,)
    # Decode byte strings to arrays of integers
    return tf.io.decode_raw(__bytes, out_type=tf.uint8, fixed_length=__dim) # (B, 4 * S)

def encode(data: any, token_size: int, sample_size: int=64) -> tf.Tensor:
    if isinstance(data, str):
        return _encode_scalar(text=data, token_size=token_size)
    else:
        return _encode_tensor(data=data, token_size=token_size, sample_size=sample_size)

# RESHAPE #####################################################################

def chunk(seq: list, size: int, repeats: bool=True) -> list:
    __chunks = (seq[__i:__i+size] for __i in range(0, len(seq), size))
    return list(__chunks if repeats else set(__chunks))

def merge(chunks: list) -> list:
    return list(itertools.chain.from_iterable(chunks))

def shape(groups: list, expand: list=[], flatten: bool=False) -> list:
    return expand + [-1] + (1 - int(flatten)) * groups

def reshape(data: tf.Tensor, groups: list, expand: list=[], flatten: bool=True) -> tf.Tensor:
    # total length of the token
    __token_size = math.prod(groups)
    # group by token unit
    __shape = shape(groups=groups, expand=expand, flatten=flatten)
    # partition or flatten the data
    return tf.reshape(tensor=data, shape=__shape) # for example (-1, G, G, G) the first dimension is not B

# AUGMENT #####################################################################

def offset(data: tf.Tensor, ticks: int=1) -> tf.Tensor:
    return tf.convert_to_tensor([ticks * b'\x00']) + data

# DECODE ######################################################################

def interpret(output: tf.Tensor) -> tf.Tensor:
    return tf.argmax(input=output, axis=-1, output_type=tf.dtypes.int32) # uint8 is not allowed

def decode(tokens: tf.Tensor) -> str:
    __b = tf.reshape(tensor=tokens, shape=(-1,)).numpy().tolist()
    return bytes(__b).decode(encoding='utf-32-be', errors='replace')

# > ###########################################################################

def preprocess(text: str, groups: list, expand: list=[], flatten: bool=True) -> tf.Tensor:
    # total length of the token
    __token_size = math.prod(groups)
    # list of bytes
    __bytes = encode(data=text, token_size=__token_size)
    # partition or flatten
    return reshape(data=__bytes, groups=groups, expand=expand, flatten=flatten)

# < ###########################################################################

def unpad(text: str) -> str:
    return text.strip('\x00')

def postprocess(output: tf.Tensor) -> str:
    # from one-hot to UTF-32 bytes
    __output = interpret(output=output)
    # flatten the groups of 4 bytes
    __output = decode(tokens=__output)
    # remove the padding
    return unpad(text=__output)

# SAMPLING ####################################################################

def sample(model: tf.keras.models.Model, text: str, **kwargs) -> tuple:
    __x = preprocess(text=text, **kwargs)
    __e = model._encoder(__x)
    __p = model._decoder(__e)
    __y = postprocess(__p)
    return (__x, __e, __p, __y)
