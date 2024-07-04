import tensorflow as tf

import tokun.pipeline

# WITH CACHE ##################################################################

class PreprocessTest(tf.test.TestCase):

    def test_shapes(self):
        __s0 = 'hello world'
        __x0 = tokun.pipeline.preprocess(text=__s0, token_size=16, expand=[])
        __x1 = tokun.pipeline.preprocess(text=__s0, token_size=16, expand=[1])
        __x2 = tokun.pipeline.preprocess(text=__s0, token_size=64, expand=[])
        self.assertEqual(tuple(__x0.shape), (48,))
        self.assertEqual(tuple(__x1.shape), (1, 48,))
        self.assertEqual(tuple(__x2.shape), (64,))

    def test_padding(self):
        __s0 = 'hello world'
        __x0 = tokun.pipeline.preprocess(text=__s0, token_size=16, expand=[])
        __x1 = tokun.pipeline.preprocess(text=__s0, token_size=64, expand=[])
        __p0 = (-len(__s0) % 4) * 4
        __p1 = (-len(__s0) % 16) * 4
        self.assertAllClose(__x0[-__p0:], tf.zeros(shape=(__p0,), dtype=tf.dtypes.uint8))
        self.assertAllClose(__x1[-__p1:], tf.zeros(shape=(__p1,), dtype=tf.dtypes.uint8))
