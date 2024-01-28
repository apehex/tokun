import tensorflow as tf

import mlable.inputs.vocabulary as _miv

# CONSTANTS ###################################################################

N_CONTEXT = 8
BLANK = '$'

# TEXT TO LIST ################################################################

def tokenize(text: str, length: int, blank=BLANK):
    __context = length * blank
    for __c in text:
        yield __context
        __context = __context[1:] + __c

# TEXT TO VECTOR ##############################################################

def dataset(text: list, stoi: callable, depth: int, context: int) -> tuple:
    __x = [_miv.encode(text=__n, stoi=stoi) for __n in tokenize(text=text, length=context)]
    __y = _miv.encode(text=text, stoi=stoi)
    return tf.convert_to_tensor(value=__x, dtype=tf.dtypes.int32), tf.one_hot(indices=__y, depth=depth, dtype=tf.dtypes.float32)
