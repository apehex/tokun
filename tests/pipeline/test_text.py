import functools

import tensorflow as tf
import tensorflow_datasets as tfds

import tokun.pipeline.text

# ENCODE ######################################################################

class EncodeDecodeCodepointsTest(tf.test.TestCase):
    def setUp(self):
        super(EncodeDecodeCodepointsTest, self).setUp()
        self._config = {'token_size': 16, 'sample_size': 4 * 1024, 'output_dtype': tf.int32}
        # specify encoding parameters
        self._encode = functools.partial(tokun.pipeline.text.encode, **self._config)
        # load the data
        self._dataset_origin = tfds.load('mlqa/en', split='test', as_supervised=False, shuffle_files=True, data_dir='~/.cache/tensorflow/', batch_size=None)
        # join the features
        self._dataset_origin = self._dataset_origin.map(lambda __x: tf.strings.join([__x['context'], __x['question']], separator='\x1d'))
        # encode the strings
        self._dataset_encoded = self._dataset_origin.map(self._encode)
        # decode back to the strings
        self._dataset_decoded = self._dataset_encoded.map(tokun.pipeline.text.decode)

    def test_dataset_specs(self):
        # >
        self.assertEqual(self._dataset_encoded.element_spec.shape, (self._config['sample_size'] // 4,))
        self.assertEqual(self._dataset_encoded.element_spec.dtype, self._config['output_dtype'])
        # <
        self.assertEqual(self._dataset_decoded.element_spec.shape, ())
        self.assertEqual(self._dataset_decoded.element_spec.dtype, tf.string)

    def test_encode_decode_reciprocal(self):
        __origin = iter(self._dataset_origin)
        __decoded = iter(self._dataset_decoded)
        for _ in range(16):
            __o = next(__origin)
            __d = next(__decoded)
            # remove the padding
            __o = __o.numpy().strip(b'\x00')
            __d = __d.numpy().strip(b'\x00')
            # check
            self.assertEqual(__o[:self._config['sample_size'] // 4], __d[:self._config['sample_size'] // 4])

    def test_specific_values(self):
        __s = 'Hello world!'
        __b = list(__s.encode('utf-8')) # utf-8 and utf-32 codepoints match on ASCII
        __x = tf.convert_to_tensor(__s, dtype=tf.string)
        __e = self._encode(__x)
        __d = tokun.pipeline.text.decode(__e)
        # ignore the padding
        self.assertEqual(__b, __e.numpy().tolist()[:len(__b)])
        self.assertEqual(__s, __d.numpy().decode('utf-8')[:len(__s)])

class EncodeDecodeBytesTest(tf.test.TestCase):
    def setUp(self):
        super(EncodeDecodeBytesTest, self).setUp()
        self._config = {'token_size': 16, 'sample_size': 4 * 1024, 'output_dtype': tf.uint8}
        # specify encoding parameters
        self._encode = functools.partial(tokun.pipeline.text.encode, **self._config)
        # load the data
        self._dataset_origin = tfds.load('mlqa/en', split='test', as_supervised=False, shuffle_files=True, data_dir='~/.cache/tensorflow/', batch_size=None)
        # join the features
        self._dataset_origin = self._dataset_origin.map(lambda __x: tf.strings.join([__x['context'], __x['question']], separator='\x1d'))
        # encode the strings
        self._dataset_encoded = self._dataset_origin.map(self._encode)
        # decode back to the strings
        self._dataset_decoded = self._dataset_encoded.map(tokun.pipeline.text.codepoint)
        self._dataset_decoded = self._dataset_decoded.map(tokun.pipeline.text.decode)

    def test_dataset_specs(self):
        # >
        self.assertEqual(self._dataset_encoded.element_spec.shape, (self._config['sample_size'],))
        self.assertEqual(self._dataset_encoded.element_spec.dtype, self._config['output_dtype'])
        # <
        self.assertEqual(self._dataset_decoded.element_spec.shape, ())
        self.assertEqual(self._dataset_decoded.element_spec.dtype, tf.string)

    def test_encode_decode_reciprocal(self):
        __origin = iter(self._dataset_origin)
        __decoded = iter(self._dataset_decoded)
        for _ in range(16):
            __o = next(__origin)
            __d = next(__decoded)
            # remove the padding
            __o = __o.numpy().strip(b'\x00')
            __d = __d.numpy().strip(b'\x00')
            # check
            self.assertEqual(__o[:self._config['sample_size'] // 4], __d[:self._config['sample_size'] // 4])

# PREPROCESSING ###############################################################

class PreprocessTest(tf.test.TestCase):

    def test_shapes(self):
        __s0 = 'hello world'
        __x0 = tokun.pipeline.text.preprocess(text=__s0, token_size=16, output_dtype=tf.uint8, expand_dims=[])
        __x1 = tokun.pipeline.text.preprocess(text=__s0, token_size=16, output_dtype=tf.uint8, expand_dims=[1])
        __x2 = tokun.pipeline.text.preprocess(text=__s0, token_size=64, output_dtype=tf.int32, expand_dims=[])
        self.assertEqual(tuple(__x0.shape), (48,))
        self.assertEqual(tuple(__x1.shape), (1, 48,))
        self.assertEqual(tuple(__x2.shape), (16,))

    def test_padding(self):
        __s0 = 'hello world'
        __x0 = tokun.pipeline.text.preprocess(text=__s0, token_size=16, output_dtype=tf.uint8, expand_dims=[])
        __x1 = tokun.pipeline.text.preprocess(text=__s0, token_size=64, output_dtype=tf.int32, expand_dims=[])
        __p0 = (-len(__s0) % 4) * 4
        __p1 = (-len(__s0) % 16)
        self.assertAllClose(__x0[-__p0:], tf.zeros(shape=(__p0,), dtype=tf.dtypes.uint8))
        self.assertAllClose(__x1[-__p1:], tf.zeros(shape=(__p1,), dtype=tf.dtypes.int32))

# POSTPROCESSING ##############################################################
