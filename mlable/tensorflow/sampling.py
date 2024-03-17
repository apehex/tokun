import tensorflow as tf

# SAMPLE ######################################################################

def _next(model: tf.Module, x: tf.Tensor) -> int:
    __prob = model(x, training=False)
    return tf.squeeze(tf.random.categorical(tf.math.log(__prob), 1)).numpy()

def sample(model: tf.Module, context: int, depth: int, length: int) -> str:
    __result = []
    __ngram = context * [0]
    for __i in range(length):
        __n = _next(model=model, x=tf.convert_to_tensor(value=[__ngram], dtype=tf.dtypes.int32))
        __ngram = __ngram[1:] + [__n]
        __result.append(__n)
    return __result
