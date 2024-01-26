import tensorflow as tf

# INITIALIZER #################################################################

class SmallNormal(tf.keras.initializers.Initializer):
    def __init__(self, mean: float=0., stddev: float=0.1):
        self._mean = 0.
        self._stddev = stddev

    def __call__(self, shape, dtype=None, **kwargs):
        return tf.random.normal(shape, mean=self._mean, stddev=self._stddev, dtype=dtype)

    def get_config(self):  # To support serialization
        return {"mean": self._mean, "stddev": self._stddev}
