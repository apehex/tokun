"""Generate random data to cover the 5 Unicode planes."""

import functools
import itertools
import math
import random

import tensorflow as tf

# UNICODE #####################################################################

def random_codepoint(lower_plane: int=0, upper_plane: int=0x40000) -> list:
    __h = '{0:0>8x}'.format(int(random.uniform(lower_plane, upper_plane)))
    return list(bytes.fromhex(__h))

def random_sample(sample_size: int, lower_plane: int=0, upper_plane: int=0x40000) -> list:
    __nested = [random_codepoint(lower_plane=lower_plane, upper_plane=upper_plane) for _ in range(sample_size)]
    return list(itertools.chain.from_iterable(__nested))

def random_dataset(size: int, sample_size: int, lower_plane: int=0, upper_plane: int=0x40000) -> tf.data.Dataset:
    def __generator() -> iter:
        for _ in range(size):
            yield random_sample(sample_size=sample_size, lower_plane=lower_plane, upper_plane=upper_plane)
    return tf.data.Dataset.from_generator(
        generator=__generator,
        output_signature=tf.TensorSpec(shape=(4 * sample_size,), dtype=tf.int32))
