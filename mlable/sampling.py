import random

import tensorflow as tf

# SAMPLE ######################################################################

def tensor(ngram: list) -> tf.Tensor:
    return tf.convert_to_tensor(value=[ngram], dtype=tf.dtypes.int32)

def _next(model: Model, x: tf.Tensor, classes: int=N_VOCABULARY, highest: bool=False) -> int:
    __prob = model(x, training=False)[0]
    __unigrams = tf.cast(x=100. * __prob, dtype=tf.dtypes.int32).numpy().tolist()
    __highest = tf.argmax(__prob, axis=-1).numpy()
    __random, _, _ = tf.nn.fixed_unigram_candidate_sampler(
        true_classes = tf.convert_to_tensor([range(N_VOCABULARY)], dtype=tf.dtypes.int64),
        num_true = classes,
        num_sampled = 1,
        unique = False,
        range_max = classes,
        unigrams = __unigrams)
    return __highest if highest else __random.numpy()[0]

def sample(model: Model, context: int=N_CONTEXT, depth: int=N_VOCABULARY, max_length: int=N_SAMPLE, itos: callable=_itos) -> str:
    __i = 0
    __start = int(random.uniform(0, N_VOCABULARY))
    __result = itos(__start)
    __ngram = (context - 1) * [0,] + [__start]
    __x = tensor(ngram=__ngram)
    __n = _next(model=model, x=__x)
    while __n != 0 and __i < max_length:
        __ngram = __ngram[1:] + [__n]
        __x = tensor(ngram=__ngram)
        __n = _next(model=model, x=__x)
        __result += itos(__n)
        __i += 1
    return __result