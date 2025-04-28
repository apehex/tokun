"""Evaluate the quality of the model output."""

import functools
import itertools
import math

import tensorflow as tf

# ACCURACY ####################################################################

def compare(left: str, right: str) -> float:
    return sum(__l == __r for __l, __r in zip(left, right)) / max(1, min(len(left), len(right)))

# TOKEN CONTENT ###############################################################

def intersection(left: str, right: str) -> float:
    __intersection = len(set(left).intersection(set(right)))
    __reference = min(len(set(left)), len(set(right)))
    return __intersection / max(1, __reference)

# ROBUSTNESS ##################################################################

def neighbors(point: tf.Tensor, radius: float, count: int) -> tf.Tensor:
    return point + tf.random.uniform(shape=(count, point.shape[-1]), minval=-radius, maxval=radius, dtype=point.dtype)
