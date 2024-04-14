import itertools
import tensorflow as tf

# GENERIC #####################################################################

def chunk(seq: list, size: int) -> list:
    return [seq[__i:__i+size] for __i in range(0, len(seq), size)]

def merge(chunks: list) -> list:
    return list(itertools.chain.from_iterable(chunks))

def context(seq: iter, length: int) -> iter:
    __context = length * [0]
    for __c in text:
        yield __context
        __context = __context[1:] + __c

# > ###########################################################################

def _tokenize_scalar(text: str) -> tf.Tensor:
    __b = tf.convert_to_tensor(value=list(text.encode('utf-32')), dtype=tf.dtypes.int32) # uint8 is not allowed
    return tf.reshape(tensor=__b, shape=(-1, 4))

def tokenize(data: tf.Tensor) -> tf.Tensor:
    # Decode bytes from UTF-8
    __bytes = tf.strings.unicode_transcode(input=data, input_encoding='UTF-8', output_encoding='UTF-32-BE')
    # Decode byte strings to arrays of integers
    __ints = tf.io.decode_raw(__bytes, out_type=tf.uint8, fixed_length=256)
    # Convert to tensor and reshape
    return tf.reshape(__ints, (-1, 4))

# < ###########################################################################

def interpret(output: tf.Tensor) -> tf.Tensor:
    return tf.argmax(input=output, axis=-1, output_type=tf.dtypes.int32) # uint8 is not allowed

def detokenize(tokens: tf.Tensor) -> str:
    __b = tf.reshape(tensor=tokens, shape=(-1,)).numpy().tolist()
    return bytes(__b).decode('utf-32-be')

# END-TO-END ##################################################################

def preprocess(dataset: tf.data.Dataset, key: str='context') -> tf.data.Dataset:
    # from UTF-8 bytes scalar to UTF-32-BE int tensor
    __dataset = dataset.map(lambda x: tokenize(x[key]))
    # one-hot encoding of UTF-32 bytes
    __dataset = __dataset.map(lambda x: tf.one_hot(indices=x, depth=256, axis=-1))
    # produce (input, target) tuples for supervised training, instead of a single tensor X
    return __dataset.map(lambda x: (x,x))

def postprocess(dataset: tf.data.Dataset) -> tf.data.Dataset:
    return dataset