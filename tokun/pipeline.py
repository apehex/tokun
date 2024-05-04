import functools
import itertools
import math

import tensorflow as tf

# GENERIC #####################################################################

def compare(left: str, right: str) -> float:
    return sum(__l == __r for __l, __r in zip(left, right)) / max(1, len(left))

def chunk(seq: list, size: int, repeats: bool=True) -> list:
    __chunks = (seq[__i:__i+size] for __i in range(0, len(seq), size))
    return list(__chunks if repeats else set(__chunks))

def merge(chunks: list) -> list:
    return list(itertools.chain.from_iterable(chunks))

def context(seq: iter, length: int) -> iter:
    __context = length * [0]
    for __c in text:
        yield __context
        __context = __context[1:] + __c

def shape(layer_count: int, group_size: int, flatten: bool=False) -> list:
    return [-1] + (1 - int(flatten)) * layer_count * [group_size]

# AUGMENT #####################################################################

def _offset(ticks: int=1, layer: int=1, unit: int=4) -> int:
    return math.ceil(ticks * (unit ** (layer - 1)))

def offset(data: tf.Tensor, ticks: int=1, layer: int=1, unit: int=4) -> tf.Tensor:
    __length = _offset(ticks=ticks, layer=layer, unit=unit)
    __pad = tf.convert_to_tensor([__length * b'\x00'])
    return __pad + data

# > ###########################################################################

def _tokenize_scalar(text: str, layer_count: int=1, group_size: int=4, flatten: bool=False) -> tf.Tensor:
    __mod = group_size ** layer_count
    __bytes = list(text.encode('utf-32-be'))
    __shape = shape(layer_count=layer_count, group_size=group_size, flatten=flatten)
    __padding = (-len(__bytes) % __mod) * [0]
    __tensor = tf.convert_to_tensor(value=__bytes + __padding, dtype=tf.dtypes.int32) # uint8 is not allowed
    return tf.reshape(tensor=__tensor, shape=__shape)

def tokenize(data: tf.Tensor, layer_count: int=1, group_size: int=4, sample_size: int=64, flatten: bool=False) -> tf.Tensor:
    # make sure each sample has a length multiple of G ** L = T, the token dim
    __mod = group_size ** layer_count
    __dim = math.ceil(4 * sample_size / __mod) * __mod # factor 4 because of the UTF-32 encoding
    # output shape
    __shape = shape(layer_count=layer_count, group_size=group_size, flatten=flatten)
    # Decode bytes from UTF-8
    __bytes = tf.strings.unicode_transcode(input=data, input_encoding='UTF-8', output_encoding='UTF-32-BE') # (B,)
    # Decode byte strings to arrays of integers
    __ints = tf.io.decode_raw(__bytes, out_type=tf.uint8, fixed_length=__dim) # (B, 4 * S)
    # group the characters into tokens
    return tf.reshape(tensor=__ints, shape=__shape) # for example (-1, G, G, G) the first dimension is not B

# < ###########################################################################

def interpret(output: tf.Tensor) -> tf.Tensor:
    return tf.argmax(input=output, axis=-1, output_type=tf.dtypes.int32) # uint8 is not allowed

def detokenize(tokens: tf.Tensor) -> str:
    __b = tf.reshape(tensor=tokens, shape=(-1,)).numpy().tolist()
    return bytes(__b).decode(encoding='utf-32-be', errors='replace')

# END-TO-END ##################################################################

def process(dataset: tf.data.Dataset, pipeline: list, replace: bool=True, feature: str=None) -> tf.data.Dataset:
    # fetch the target feature in the dataset
    __dataset = dataset.map(lambda x: x[feature]) if feature else dataset
    # specify how to combine each operation result with the original dataset
    __replace = len(list(pipeline)) * [replace] if isinstance(replace, bool) else replace
    # apply the operation successively  
    for __fn, __repl in zip(pipeline, __replace):
        __new = __dataset.map(__fn)
        __dataset = __new if __repl else __dataset.concatenate(__new)
    return __dataset

def postprocess(output: tf.Tensor) -> tf.Tensor:
    # from one-hot to UTF-32 bytes
    __output = interpret(output=output)
    # flatten the groups of 4 bytes
    return detokenize(tokens=__output)
