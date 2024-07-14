"""Generate random data to cover the 5 Unicode planes."""

import functools
import itertools
import math
import random

import tensorflow as tf

# UNICODE #####################################################################

def _random_codepoint_binary(lower_plane: int=0, upper_plane: int=0x40000) -> list:
    __h = '{0:0>32b}'.format(int(random.uniform(lower_plane, upper_plane)))
    return [int(__b) for __b in __h]

def _random_codepoint_bytes(lower_plane: int=0, upper_plane: int=0x40000) -> list:
    __h = '{0:0>8x}'.format(int(random.uniform(lower_plane, upper_plane)))
    return list(bytes.fromhex(__h))

def random_codepoint(lower_plane: int=0, upper_plane: int=0x40000, binary: bool=False) -> list:
    return _random_codepoint_binary(lower_plane=lower_plane, upper_plane=upper_plane) if binary else _random_codepoint_bytes(lower_plane=lower_plane, upper_plane=upper_plane)

def random_sample(sample_size: int, lower_plane: int=0, upper_plane: int=0x40000, binary: bool=False) -> list:
    __nested = [random_codepoint(lower_plane=lower_plane, upper_plane=upper_plane, binary=binary) for _ in range(sample_size)]
    return list(itertools.chain.from_iterable(__nested))

def random_dataset(size: int, sample_size: int, lower_plane: int=0, upper_plane: int=0x40000, binary: bool=False) -> tf.data.Dataset:
    __factor = 32 if binary else 4
    # sample generator
    def __generator() -> iter:
        for _ in range(size):
            yield random_sample(sample_size=sample_size, lower_plane=lower_plane, upper_plane=upper_plane, binary=binary)
    # wrap into a dataset
    return tf.data.Dataset.from_generator(
        generator=__generator,
        output_signature=tf.TensorSpec(shape=(__factor * sample_size,), dtype=tf.int32))
