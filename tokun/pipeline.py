"""Pre and post processing pipelines for tokun."""

import functools
import itertools
import math

import keras as ks
import tensorflow as tf

# UNICODE #####################################################################

CODE_STX = b'\x02'
CODE_ETX = b'\x03'
CODE_FS = b'\x1c'
CODE_GS = b'\x1d'
CODE_RS = b'\x1e'
CODE_US = b'\x1f'

# ENCODE ######################################################################

def encode(data: tf.Tensor, token_size: int, sample_size: int) -> tf.Tensor:
    # factor 4 because of the UTF-32 encoding
    __dim = math.ceil(4 * sample_size / token_size) * token_size
    # Decode bytes from UTF-8
    __bytes = tf.strings.unicode_transcode(input=data, input_encoding='UTF-8', output_encoding='UTF-32-BE') # (B,)
    # Decode byte strings to arrays of integers
    return tf.io.decode_raw(__bytes, out_type=tf.uint8, fixed_length=__dim) # (B, 4 * S)

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

def interpret(output: tf.Tensor) -> tf.Tensor:
    return tf.argmax(input=output, axis=-1, output_type='int32') # uint8 is not allowed

def decode(tokens: tf.Tensor) -> str:
    __b = tf.reshape(tensor=tokens, shape=(-1,)).numpy().tolist()
    return bytes(__b).decode(encoding='utf-32-be', errors='replace')

# > ###########################################################################

def preprocess(text: str, token_size: int, expand: list=[]) -> tf.Tensor:
    # as tensor
    __data = tf.convert_to_tensor(text, dtype='string')
    # list of bytes
    __bytes = encode(data=__data, token_size=token_size, sample_size=len(text))
    # partition or flatten
    return reshape(data=__bytes, expand=expand)

# < ###########################################################################

def unpad(text: str) -> str:
    return text.strip('\x00')

def postprocess(output: tf.Tensor, onehot: bool=True) -> str:
    # from one-hot to UTF-32 bytes
    __output = interpret(output=output) if onehot else output
    # flatten the groups of 4 bytes
    __output = decode(tokens=__output)
    # remove the padding
    return unpad(text=__output)

# SAMPLING ####################################################################

def sample(model: ks.models.Model, text: str, **kwargs) -> tuple:
    __x = preprocess(text=text, **kwargs)
    __e = model._encoder(__x)
    __p = model._decoder(__e)
    __y = postprocess(__p)
    return (__x, __e, __p, __y)
