import random

import tensorflow as tf

# SAMPLE ######################################################################

def _next(model: tf.Module, x: tf.Tensor) -> int:
    __prob = tf.squeeze(model(x, training=False))
    return tf.argmax(__prob, axis=-1).numpy()

def sample(model: tf.Module, context: int, depth: int, length: int, itos: callable) -> str:
    __start = int(random.uniform(0, depth))
    __result = itos(__start)
    __ngram = (context - 1) * [0,] + [__start]
    __x = tf.convert_to_tensor(value=[__ngram], dtype=tf.dtypes.int32)
    print(__ngram)
    print(__x)
    __n = _next(model=model, x=__x)
    for __i in range(length):
        __ngram = __ngram[1:] + [__n]
        __x = tf.convert_to_tensor(value=[__ngram], dtype=tf.dtypes.int32)
        __n = _next(model=model, x=__x)
        __result += itos(__n)
    return __result
