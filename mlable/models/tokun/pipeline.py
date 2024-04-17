import itertools
import math

import tensorflow as tf

# GENERIC #####################################################################

def chunk(seq: list, size: int) -> list:
    return [seq[__i:__i+size] for __i in range(0, len(seq), size)]

def merge(chunks: list) -> list:
    return list(itertools.chain.from_iterable(chunks))

def context(seq: iter, length: int) -> iter:
    __context = length * [0]
    for __c in text:
        yield __context
        __context = __context[1:] + __c

# > ###########################################################################

def _tokenize_scalar(text: str, layer_count: int=1, group_size: int=4) -> tf.Tensor:
    __mod = group_size ** layer_count
    __bytes = list(text.encode('utf-32'))
    __padding = (-len(__bytes) % __mod) * [0]
    __tensor = tf.convert_to_tensor(value=__bytes + __padding, dtype=tf.dtypes.int32) # uint8 is not allowed
    return tf.reshape(tensor=__tensor, shape=[-1] + layer_count * [group_size])

def tokenize(data: tf.Tensor, layer_count: int=1, group_size: int=4, sample_size: int=64) -> tf.Tensor:
    # TODO: split each sample into chunks of length G ** L = T, the token dim, and reshape so that there's one token / row
    __mod = group_size ** layer_count
    __dim = math.ceil(4 * sample_size / __mod) * __mod
    # Decode bytes from UTF-8
    __bytes = tf.strings.unicode_transcode(input=data, input_encoding='UTF-8', output_encoding='UTF-32-BE') # (B, )
    # Decode byte strings to arrays of integers
    __ints = tf.io.decode_raw(__bytes, out_type=tf.uint8, fixed_length=__dim) # (B, 4 * S)
    # group the characters into tokens
    return tf.reshape(tensor=__ints, shape=[-1] + layer_count * [group_size]) # (-1, G, G, G) the first dimension is not B

# < ###########################################################################

def interpret(output: tf.Tensor) -> tf.Tensor:
    return tf.argmax(input=output, axis=-1, output_type=tf.dtypes.int32) # uint8 is not allowed

def detokenize(tokens: tf.Tensor) -> str:
    __b = tf.reshape(tensor=tokens, shape=(-1,)).numpy().tolist()
    return bytes(__b).decode('utf-32-be')

# END-TO-END ##################################################################

def preprocess(dataset: tf.data.Dataset, key: str='context', layer_count: int=1, group_size: int=4, sample_size: int=64) -> tf.data.Dataset:
    # from UTF-8 bytes scalar to UTF-32-BE int tensor
    __dataset = dataset.map(lambda x: tokenize(data=x[key], layer_count=layer_count, group_size=group_size, sample_size=sample_size))
    # one-hot encoding of UTF-32 bytes
    __dataset = __dataset.map(lambda x: tf.one_hot(indices=x, depth=256, axis=-1))
    # produce (input, target) tuples for supervised training, instead of a single tensor X
    return __dataset.map(lambda x: (x,x))

def postprocess(output: tf.Tensor) -> tf.Tensor:
    # from one-hot to UTF-32 bytes
    __output = interpret(output=output)
    # flatten the groups of 4 bytes
    return detokenize(tokens=__output)
