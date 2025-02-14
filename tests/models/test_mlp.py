import functools
import math

import tensorflow as tf

import tokun.models.mlp
import tokun.pipeline.text

# ENCODE ######################################################################

class AutoEncoderTest(tf.test.TestCase):
    def setUp(self):
        super(AutoEncoderTest, self).setUp()
        # model configs
        self._configs = {
            'binary': {
                'token_dim': 4,
                'latent_dim': 8,
                'input_dim': 256,
                'embed_dim': 4,
                'output_dim': 8,
                'sequence_axis': 1,
                'feature_axis': -1,
                'activation': 'gelu',},}
        # raw text
        self._cases = [
            'hello world',]
        # preprocessing args
        self._args = {
            'token_dim': 4 * 128, # oversized to ensure a constant seq dim
            'expand_dims': [1],
            'encode_dtype': tf.uint8,
            'output_dtype': tf.int32,}
        # custom preprocessing
        self._preprocess = functools.partial(tokun.pipeline.text.preprocess, **self._args)

    def test_shapes(self):
        for __config in self._configs.values():
            # init
            __model = tokun.models.mlp.AutoEncoder(**__config)
            # use the model's config, since it is normalized (token and latent dims are both lists)
            __token_dim, __latent_dim, __output_dim = math.prod(__model._config['token_dim']), __model._config['latent_dim'][-1], __model._config['output_dim']
            __batch_dim, __sequence_dim = (1, self._args['token_dim'])
            # build
            __model.build((__batch_dim, __sequence_dim))
            for __case in self._cases:
                __token_val = self._preprocess(__case)
                __embed_val = __model.encode(__token_val)
                __output_val = __model.decode(__embed_val)
                self.assertEqual((__batch_dim, __sequence_dim // __token_dim, __latent_dim), tuple(__embed_val.shape))
                self.assertEqual((__batch_dim, __sequence_dim, __output_dim), tuple(__output_val.shape))

    def test_dtypes(self):
        for __config in self._configs.values():
            # init
            __model = tokun.models.mlp.AutoEncoder(**__config)
            # use the model's config, since it is normalized (token and latent dims are both lists)
            __token_dim, __latent_dim, __output_dim = math.prod(__model._config['token_dim']), __model._config['latent_dim'][-1], __model._config['output_dim']
            __batch_dim, __sequence_dim = (1, self._args['token_dim'])
            # build
            __model.build((__batch_dim, __sequence_dim))
            for __case in self._cases:
                __token_val = self._preprocess(__case)
                __embed_val = __model.encode(__token_val)
                __output_val = __model.decode(__embed_val)
                self.assertEqual(__embed_val.dtype, tf.float32)
                self.assertEqual(__output_val.dtype, tf.float32)
