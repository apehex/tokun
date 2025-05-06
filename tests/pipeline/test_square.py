import tensorflow as tf

import mlable.sampling
import mlable.text
import tokun.data
import tokun.eval
import tokun.pipeline.square.postprocess
import tokun.pipeline.square.preprocess

# CONSTANTS ####################################################################

DATASET = tokun.data.random_dataset_of_strings(sample_count=128, sample_size=1024)

SAMPLES = [
    """      ,~.\r\n   ,-'__ `-,\r\n  {,-'  `. }              ,')\r\n ,( a )   `-.__         ,',')~,\r\n<=.) (         `-.__,==' ' ' '}\r\n  (   )                      /\r\n   `-'\\   ,                  )\r\n       |  \\        `~.      /\r\n       \\   `._        \\    /\r\n        \\     `._____,'   /\r\n         `-.            ,'\r\n            `-.      ,-'\r\n               `~~~~'\r\n               //_||\r\n            __//--'/`   hjw\r\n          ,--'/`  '\r\n             '\r""",
    """ /\\___/\\\r\n \\/   \\/\r\n  \\~ ~/\r\n ==`^ ==\r\n  /   \\        |\\___/|\r\n /|   |        \\/- -\\/ ____...,...\r\n || - |         \\o o/             \\\r\n ||   |        ==`^ ==   ,        /\\\r\n ||| ||_            `.  / --- \\  / \\\\____//\r\n/\\||_|//         ;____,'      | /   ` ---\r\n\\_____/                    ;___/\r""",
    """⢕⢕⢕⢕⢕⢕⢕⢕⢕⢕⢕⢕⢕⢕⢕⢕⢕⢕⢕⢕⢕⢕⠕⠕⠕⠕⢕⢕\n⢕⢕⢕⢕⢕⠕⠕⢕⢕⢕⢕⢕⢕⢕⢕⢕⢕⠕⠁⣁⣠⣤⣤⣤⣶⣦⡄⢑\n⢕⢕⢕⠅⢁⣴⣤⠀⣀⠁⠑⠑⠁⢁⣀⣀⣀⣀⣘⢻⣿⣿⣿⣿⣿⡟⢁⢔\n⢕⢕⠕⠀⣿⡁⠄⠀⣹⣿⣿⣿⡿⢋⣥⠤⠙⣿⣿⣿⣿⣿⡿⠿⡟⠀⢔⢕\n⢕⠕⠁⣴⣦⣤⣴⣾⣿⣿⣿⣿⣇⠻⣇⠐⠀⣼⣿⣿⣿⣿⣿⣄⠀⠐⢕⢕\n⠅⢀⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⣶⣶⣿⣿⣿⣿⣿⣿⣿⣿⣷⡄⠐⢕\n⠀⢸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡄⠐\n⢄⠈⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡆\n⢕⢔⠀⠈⠛⠿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿\n⢕⢕⢄⠈⠳⣶⣶⣶⣤⣤⣤⣤⣭⡍⢭⡍⢨⣯⡛⢿⣿⣿⣿⣿⣿⣿⣿⣿\n⢕⢕⢕⢕⠀⠈⠛⠿⢿⣿⣿⣿⣿⣿⣦⣤⣿⣿⣿⣦⣈⠛⢿⢿⣿⣿⣿⣿\n⢕⢕⢕⠁⢠⣾⣶⣾⣭⣖⣛⣿⠿⣿⣿⣿⣿⣿⣿⣿⣿⣷⡆⢸⣿⣿⣿⡟\n⢕⢕⠅⢀⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠟⠈⢿⣿⣿⡇\n⢕⠕⠀⠼⠟⢉⣉⡙⠻⠿⢿⣿⣿⣿⣿⣿⡿⢿⣛⣭⡴⠶⠶⠂⠀⠿⠿⠇""",]

# PREPROCESS ###################################################################

class PreprocessTests(tf.test.TestCase):
    def setUp(self):
        super(PreprocessTests, self).setUp()
        self._cases = [
            {
                'args': {
                    'batch_dim': 1,
                    'height_dim': 32,
                    'width_dim': 64,
                    'token_dim': 4,
                    'drop_dim': 0,
                    'encoding': 'UTF-8',
                    'features': [],},
                'shapes': {
                    'inputs': [1, 32, 64 // 4, 4],
                    'targets': [1, 32, 64 // 4, 32],},},
            {
                'args': {
                    'batch_dim': 1,
                    'height_dim': 32,
                    'width_dim': 64,
                    'token_dim': 1,
                    'drop_dim': 0,
                    'encoding': 'UTF-8',
                    'features': [],},
                'shapes': {
                    'inputs': [1, 32, 64, 1],
                    'targets': [1, 32, 64, 8],},},
            {
                'args': {
                    'batch_dim': 1,
                    'height_dim': 32,
                    'width_dim': 64,
                    'token_dim': 6,
                    'drop_dim': 1,
                    'encoding': 'UTF-32-BE',
                    'features': [],},
                'shapes': {
                    'inputs': [1, 32, 64 // 8, 6],
                    'targets': [1, 32, 64 // 8, 48],},},
            {
                'args': {
                    'batch_dim': 1,
                    'height_dim': 32,
                    'width_dim': 64,
                    'token_dim': 1,
                    'drop_dim': 1,
                    'encoding': 'UTF-32-BE',
                    'features': [],},
                'shapes': {
                    'inputs': [1, 32, 3 * (64 // 4), 1],
                    'targets': [1, 32, 3 * (64 // 4), 8],},},]

    def test_shape_and_dtype(self):
        for __case in self._cases:
            __preprocess = tokun.pipeline.square.preprocess.factory(**__case['args'])
            for __sample in SAMPLES:
                __s = tf.cast([__sample], dtype=tf.string)
                __x, __t = __preprocess(__s)
                # shapes
                self.assertEqual(tuple(__x.shape), tuple(__case['shapes']['inputs']))
                self.assertEqual(tuple(__t.shape), tuple(__case['shapes']['targets']))
                # shapes
                self.assertEqual(__x.dtype, tf.uint8)
                self.assertEqual(__t.dtype, tf.float32)

    def test_on_dataset_batches(self):
        __dataset = iter(DATASET.batch(16))
        __preprocess = tokun.pipeline.square.preprocess.factory(batch_dim=16, height_dim=32, width_dim=128, token_dim=4, drop_dim=0, encoding='UTF-8', features=[],)
        for _ in range(4):
            __s = next(__dataset)
            __x, __t = __preprocess(__s)
            # shapes
            self.assertEqual(tuple(__x.shape), (16, 32, 32, 4))
            self.assertEqual(tuple(__t.shape), (16, 32, 32, 32))
            # shapes
            self.assertEqual(__x.dtype, tf.uint8)
            self.assertEqual(__t.dtype, tf.float32)

    def test_inputs_equal_targets(self):
        for __case in self._cases:
            __preprocess = tokun.pipeline.square.preprocess.factory(**__case['args'])
            for __sample in SAMPLES:
                __s = tf.cast([__sample], dtype=tf.string)
                __x, __t = __preprocess(__s)
                self.assertAllEqual(__x, mlable.sampling.binary(__t, depth=8, threshold=0.5, dtype=tf.int32))

    def test_preprocess_postprocess_reciprocity(self):
        for __case in self._cases:
            __preprocess = tokun.pipeline.square.preprocess.factory(**__case['args'])
            __postprocess = tokun.pipeline.square.postprocess.factory(drop_dim=__case['args']['drop_dim'], encoding=__case['args']['encoding'], threshold=0.5, errors='ignore')
            for __sample in SAMPLES:
                __s = tf.cast([__sample], dtype=tf.string)
                __x, __t = __preprocess(__s)
                __o = __postprocess(__t).numpy().tolist()
                __o = b'\n'.join(__o[0]).decode('UTF-8', errors='ignore')
                assert int(tokun.eval.compare(__sample, __o[0])) == 1
