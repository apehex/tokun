import random

import tensorflow as tf

# SAMPLE ######################################################################

def tensor(ngram: list) -> tf.Tensor:
    return tf.convert_to_tensor(value=[ngram], dtype=tf.dtypes.int32)

def _next(model: tf.Module, x: tf.Tensor, classes: int, highest: bool=False) -> int:
    __prob = model(x, training=False)[0]
    __unigrams = tf.cast(x=100. * __prob, dtype=tf.dtypes.int32).numpy().tolist()
    __highest = tf.argmax(__prob, axis=-1).numpy()
    __random, _, _ = tf.nn.fixed_unigram_candidate_sampler(
        true_classes = tf.convert_to_tensor([range(classes)], dtype=tf.dtypes.int64),
        num_true = classes,
        num_sampled = 1,
        unique = False,
        range_max = classes,
        unigrams = __unigrams)
    return __highest if highest else __random.numpy()[0]

def sample(model: tf.Module, context: int, depth: int, max_length: int, itos: callable) -> str:
    __i = 0
    __start = int(random.uniform(0, depth))
    __result = itos(__start)
    __ngram = (context - 1) * [depth,] + [__start]
    __x = tensor(ngram=__ngram)
    __n = _next(model=model, x=__x, classes=depth, highest=False)
    while __n != 0 and __i < max_length:
        __ngram = __ngram[1:] + [__n]
        __x = tensor(ngram=__ngram)
        __n = _next(model=model, x=__x, classes=depth, highest=False)
        __result += itos(__n)
        __i += 1
    return __result
