import tensorflow as tf

import mlable.sampling
import mlable.text
import tokun.data
import tokun.eval
import tokun.pipeline.flat.postprocess
import tokun.pipeline.flat.preprocess

# CONSTANTS ####################################################################

DATASET = tokun.data.random_dataset_of_strings(sample_count=128, sample_size=256)

SAMPLES = [
    """위키백과, 우리 모두의 백과사전.\nt-분포 확률적 임베딩(t-SNE)은 데이터의 차원 축소에 사용되는 기계 학습 알고리즘 중 하나로, 2002년 샘 로이스Sam Rowise와 제프리 힌튼에 의해 개발되었다.[1] t-SNE는 비선형 차원 축소 기법으로, 고차원 데이터를 특히 2, 3차원 등으로 줄여 가시화하는데에 유용하게 사용된다. 구체적으로 t-SNE는 비슷한 데이터는 근접한 2, 3차원의 지점으로, 다른 데이터는 멀리 떨어진 지점으로 맵핑한다.""",
    """class Encoder(tf.keras.models.Model):\n    def __init__(self, depth: int, token_dim: int, encoding_dim: int, embedding_dim: int, batch_dim: int=None, attention: bool=False, **kwargs) -> None:\n        super(Encoder, self).__init__(**kwargs)\n        self._encoder = tf.keras.Sequential([\n            tf.keras.Input(shape=(encoding_dim,), batch_size=batch_dim, name='input'), # (B * G ^ D, U)\n            tf.keras.layers.Dense(units=embedding_dim, activation=None, use_bias=False, kernel_initializer='glorot_uniform', bias_initializer=None, name='embed-1'),] # (B * G ^ D, U) => (B * G ^ D, E)\n            + [tokun.layers.TokenizeBlock(left_axis=-2, right_axis=-1, token_dim=token_dim, attention=attention, name='tokenize' + (__i + 1) * '-4') for __i in range(depth)]) # (B * G ^ i, E) => (B * G ^ (i-1), E)\n\n    def call(self, x: tf.Tensor) -> tf.Tensor:\n        return self._encoder(x)\n""",
    """Hilbert curve\n\nThe Hilbert curve (also known as the Hilbert space-filling curve) is a continuous fractal space-filling curve first described by the German mathematician David Hilbert in 1891,[1] as a variant of the space-filling Peano curves discovered by Giuseppe Peano in 1890.[2]\n\nBecause it is space-filling, its Hausdorff dimension is 2 (precisely, its image is the unit square, whose dimension is 2 in any definition of dimension; its graph is a compact set homeomorphic to the closed unit interval, with Hausdorff dimension 1).\n\nThe Hilbert curve is constructed as a limit of piecewise linear curves. The length of the {\\displaystyle n}th curve is {\\displaystyle \\textstyle 2^{n}-{1 \\over 2^{n}}}, i.e., the length grows exponentially with {\\displaystyle n}, even though each curve is contained in a square with area {\\displaystyle 1}.\n\nImages\n\nFirst six iterations of the Hilbert curve\n\nHilbert curve, first order\n\nHilbert curves, first and second orders\n\nHilbert curves, first to third orders\n\nProduction rules\n\nHilbert curve, construction color-coded\n\nA 3-D Hilbert curve with color showing progression\n\nVariant, first three iterations[3]\n\nApplications and mapping algorithms\n\nBoth the true Hilbert curve and its discrete approximations are useful because they give a mapping between 1D and 2D space that preserves locality fairly well.[4] This means that two data points which are close to each other in one-dimensional space are also close to each other after folding. The converse cannot always be true.\n\nBecause of this locality property, the Hilbert curve is widely used in computer science. For example, the range of IP addresses used by computers can be mapped into a picture using the Hilbert curve. Code to generate the image would map from 2D to 1D to find the color of each pixel, and the Hilbert curve is sometimes used because it keeps nearby IP addresses close to each other in the picture.[5] The locality property of the Hilbert curve has also been used to design algorithms for exploring regions with mobile robots[6][7] and indexing geospatial location data.[8]\n\nIn an algorithm called Riemersma dithering, grayscale photographs can be converted to a dithered black-and-white image using thresholding, with the leftover amount from each pixel added to the next pixel along the Hilbert curve. Code to do this would map from 1D to 2D, and the Hilbert curve is sometimes used because it does not create the distracting patterns that would be visible to the eye if the order were simply left to right across each row of pixels.[9] Hilbert curves in higher dimensions are an instance of a generalization of Gray codes, and are sometimes used for similar purposes, for similar reasons. For multidimensional databases, Hilbert order has been proposed to be used instead of Z order because it has better locality-preserving behavior. For example, Hilbert curves have been used to compress and accelerate R-tree indexes[10] (see Hilbert R-tree). They have also been used to help compress data warehouses.[11][12]\n\nThe linear distance of any point along the curve can be converted to coordinates in n dimensions for a given n, and vice versa, using any of several standard mathematical techniques such as Skilling\'s method.[13][14]\n\nIt is possible to implement Hilbert curves efficiently even when the data space does not form a square.[15] Moreover, there are several possible generalizations of Hilbert curves to higher dimensions.[16][17]\n\nRepresentation as Lindenmayer system\n\nThe Hilbert Curve can be expressed by a rewrite system (L-system).\n\nDuration: 52 seconds.0:52\nHilbert curve at its sixth iteration\nAlphabet : A, B\nConstants : F + −\nAxiom : A\nProduction rules:\nA → +BF−AFA−FB+\nB → −AF+BFB+FA−\nHere, "F" means "draw forward", "+" means "turn left 90°", "-" means "turn right 90°" (see turtle graphics), and "A" and "B" are ignored during drawing.\n\nOther implementations\n\nGraphics Gems II[18][promotion?] discusses Hilbert curve coherency, and provides implementation.\n\nThe Hilbert Curve is commonly used among rendering images or videos. Common programs such as Blender and Cinema 4D use the Hilbert Curve to trace the objects, and render the scene.[citation needed]\n\nThe slicer software used to convert 3D models into toolpaths for a 3D printer typically has the Hilbert curve as an option for an infill pattern.\n""",
    """Vícerozměrná náhodná proměnná nebo náhodný vektor je v teorii pravděpodobnosti a statistice seznam matematických proměnných, jehož žádná hodnota není známa, buď protože zatím nebyla pozorována, nebo protože její hodnotu neznáme přesně. Jednotlivé proměnné jsou sdružené v náhodném vektoru, protože tvoří části jednoho matematického systému – často reprezentují různé vlastnosti určité statistické jednotky. Pokud například chceme zachytit, že každá osoba má určitý věk, výšku a hmotnost, lze tyto vlastnosti blíže neurčené osoby z určité skupiny reprezentovat náhodným vektorem. Prvky náhodných vektorů jsou obvykle reálná čísla.""",]

# PREPROCESS ###################################################################

class PreprocessTests(tf.test.TestCase):
    def setUp(self):
        super(PreprocessTests, self).setUp()
        self._cases = [
            {
                'args': {
                    'batch_dim': 1,
                    'sample_dim': 128,
                    'token_dim': 4,
                    'drop_dim': 0,
                    'encoding': 'UTF-8',
                    'features': [],},
                'shapes': {
                    'inputs': [1, 128 // 4, 4],
                    'targets': [1, 128 // 4, 32],},},
            {
                'args': {
                    'batch_dim': 1,
                    'sample_dim': 128,
                    'token_dim': 1,
                    'drop_dim': 0,
                    'encoding': 'UTF-8',
                    'features': [],},
                'shapes': {
                    'inputs': [1, 128, 1],
                    'targets': [1, 128, 8],},},
            {
                'args': {
                    'batch_dim': 1,
                    'sample_dim': 128,
                    'token_dim': 6,
                    'drop_dim': 1,
                    'encoding': 'UTF-32-BE',
                    'features': [],},
                'shapes': {
                    'inputs': [1, 128 // 8, 6],
                    'targets': [1, 128 // 8, 48],},},
            {
                'args': {
                    'batch_dim': 1,
                    'sample_dim': 128,
                    'token_dim': 1,
                    'drop_dim': 1,
                    'encoding': 'UTF-32-BE',
                    'features': [],},
                'shapes': {
                    'inputs': [1, 3 * (128 // 4), 1],
                    'targets': [1, 3 * (128 // 4), 8],},},]

    def test_shape_and_dtype(self):
        for __case in self._cases:
            __preprocess = tokun.pipeline.flat.preprocess.factory(**__case['args'])
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
        __preprocess = tokun.pipeline.flat.preprocess.factory(batch_dim=16, sample_dim=128, token_dim=4, drop_dim=0, encoding='UTF-8', features=[],)
        for _ in range(4):
            __s = next(__dataset)
            __x, __t = __preprocess(__s)
            # shapes
            self.assertEqual(tuple(__x.shape), (16, 32, 4))
            self.assertEqual(tuple(__t.shape), (16, 32, 32))
            # shapes
            self.assertEqual(__x.dtype, tf.uint8)
            self.assertEqual(__t.dtype, tf.float32)

    def test_inputs_equal_targets(self):
        for __case in self._cases:
            __preprocess = tokun.pipeline.flat.preprocess.factory(**__case['args'])
            for __sample in SAMPLES:
                __s = tf.cast([__sample], dtype=tf.string)
                __x, __t = __preprocess(__s)
                self.assertAllEqual(__x, mlable.sampling.binary(__t, depth=8, threshold=0.5, dtype=tf.int32))

    def test_preprocess_postprocess_reciprocity(self):
        for __case in self._cases:
            __preprocess = tokun.pipeline.flat.preprocess.factory(**__case['args'])
            __postprocess = tokun.pipeline.flat.postprocess.factory(drop_dim=__case['args']['drop_dim'], encoding=__case['args']['encoding'], threshold=0.5, errors='ignore')
            for __sample in SAMPLES:
                __s = tf.cast([__sample], dtype=tf.string)
                __x, __t = __preprocess(__s)
                __o = mlable.text.unpack(__postprocess(__t))
                assert int(tokun.eval.compare(__sample, __o[0])) == 1
