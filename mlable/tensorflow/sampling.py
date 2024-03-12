import random

import tensorflow as tf

# SAMPLE ######################################################################

def _next(model: tf.Module, x: tf.Tensor) -> int:
    __prob = tf.squeeze(model(x, training=False))
    return tf.argmax(__prob, axis=-1).numpy()

def sample(model: tf.Module, context: int, depth: int, length: int) -> str:
    __result = [int(random.uniform(0, depth))]
    __ngram = (context - 1) * [0,] + __result
    for __i in range(length):
        __x = tf.convert_to_tensor(value=[__ngram], dtype=tf.dtypes.int32)
        __n = _next(model=model, x=__x)
        __ngram = __ngram[1:] + [__n]
        __result.append(__n)
    return __result
