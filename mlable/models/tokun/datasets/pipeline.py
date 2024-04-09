import itertools
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

def tokenize(text: str) -> tf.Tensor:
    __b = tf.convert_to_tensor(value=list(text.encode('utf-32')), dtype=tf.dtypes.int32) # uint8 is not allowed
    return tf.reshape(tensor=__b, shape=(-1, 4))

# < ###########################################################################

def interpret(output: tf.Tensor) -> tf.Tensor:
    return tf.argmax(input=output, axis=-1, output_type=tf.dtypes.int32) # uint8 is not allowed

def detokenize(tokens: tf.Tensor) -> str:
    __b = tf.reshape(tensor=tokens, shape=(-1,)).numpy().tolist()
    return bytes(__b).decode('utf-32')
