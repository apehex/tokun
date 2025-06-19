import functools
import math

import tensorflow as tf

import mlable.sampling
import mlable.shaping.axes
import mlable.shaping.hilbert
import mlable.text

# SAMPLING #####################################################################

def _sampler_factory(threshold: float=0.0, temp: float=1.0, topp: float=-1.0, topk: int=-1, binary: bool=True, bigendian: bool=True) -> callable:
    __fn = mlable.sampling.binary if binary else mlable.sampling.categorical
    # common args
    __args = {'temp': temp, 'topp': topp, 'topk': topk, 'depth': 8 if binary else 256, 'dtype': tf.int32}
    # binary args
    if binary:
        __args['bigendian'] = bigendian
        __args['threshold'] = threshold
    # compound individual probabilities into byte probabilities
    __bytes = functools.partial(__fn, **__args)
    # sample from logits
    def __sampler(outputs: tf.Tensor) -> tf.Tensor:
        return __bytes(outputs)
    # customized fn
    return __sampler

# FORMATTING ###################################################################

def _formatter_factory(order_num: int, rank_num: int) -> callable:
    # folding created rank spatial axes
    __axes = list(range(1, rank_num + 1))
    # unfold all the hilbert axes
    __unfold = functools.partial(mlable.shaping.hilbert.unfold, order=order_num, rank=rank_num, axes=__axes)
    # merge the token axis
    __merge = functools.partial(mlable.shaping.axes.merge, axis=-1, right=False)
    # merge all the spatial axes
    def __formatter(outputs: tf.Tensor) -> tf.Tensor:
        return __merge(__unfold(outputs))
    # customized fn
    return __formatter

# DECODING #####################################################################

def _decoder_factory(encoding: str='UTF-32-BE', errors: str='replace') -> callable:
    # decode the sequence of bytes into strings
    __string = functools.partial(mlable.text.decode, encoding=encoding, errors=errors)
    # from bytes to characters
    def __decoder(outputs: tf.Tensor) -> tf.Tensor:
        return __string(outputs)
    # customized fn
    return __decoder

# > END-TO-END #################################################################

def _wrapper(outputs: tf.Tensor, sampler: callable, formatter: callable, decoder: callable, cleaner: callable) -> tuple: # masker: callable
    # sample according to the probabilities / logits
    __outputs = sampler(outputs)
    # merge the spatial dimensions
    __outputs = formatter(__outputs)
    # decode the byte values into text
    __outputs = decoder(__outputs)
    # get rid of the padding
    return cleaner(__outputs)

def factory(order_num: int, rank_num: int, threshold: float=0.0, temp: float=1.0, topp: float=-1.0, topk: int=-1, binary: bool=True, bigendian: bool=True, encoding: str='UTF-32-BE', errors: str='replace') -> callable:
    # custom fn
    __sampler = _sampler_factory(threshold=threshold, temp=temp, topp=topp, topk=topk, binary=binary, bigendian=bigendian)
    __formatter = _formatter_factory(order_num=order_num, rank_num=rank_num)
    __decoder = _decoder_factory(encoding=encoding, errors=errors)
    # actual preprocessing function
    return functools.partial(_wrapper, sampler=__sampler, formatter=__formatter, decoder=__decoder, cleaner=mlable.text.unpad)
