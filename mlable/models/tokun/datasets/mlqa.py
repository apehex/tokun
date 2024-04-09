import tensorflow as tf

#

def preprocess(batch: tf.Tensor) -> str:
    __flat = tf.reshape(tensor=batch, shape=(-1,))
    __concat = b''.join(__flat.numpy().tolist())
    return __concat.decode('utf-8')

#
