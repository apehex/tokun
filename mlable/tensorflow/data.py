import tensorflow as tf

# TEXT TO VECTOR ##############################################################

def dataset(x: list, y: list, depth: int) -> tuple:
    return tf.constant(tf.convert_to_tensor(value=x, dtype=tf.dtypes.int32)), tf.constant(tf.one_hot(indices=y, depth=depth, dtype=tf.dtypes.float32))
