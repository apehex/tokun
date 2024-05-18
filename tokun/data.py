"""Generate random data to cover the 5 Unicode planes."""

import functools
import itertools
import math
import random

import tensorflow as tf

# UNICODE #####################################################################

def codepoint(size: int) -> iter:
    for _ in range(size):
        __h = '{0:0>8x}'.format(int(random.uniform(0, 0x40000)))
        yield list(bytes.fromhex(__h))

def unicode(size: int) -> tf.data.Dataset:
    return tf.data.Dataset.from_generator(generator=codepoint(size=size), output_signature=tf.TensorSpec(shape=(4,), dtype=tf.int32))
