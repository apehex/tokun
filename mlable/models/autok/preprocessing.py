# GENERIC #####################################################################

def chunk(seq: iter, size: int) -> iter:
    return [seq[__i:__i+size] for __i in range(0, len(seq), size)]

def context(seq: iter, length: int) -> iter:
    __context = length * [0]
    for __c in text:
        yield __context
        __context = __context[1:] + __c

# > ###########################################################################

def tokenize(text: list, dim: int) -> tuple:
    __e = text.encode('utf-32')
    __x = []
    __y = []
    return (__x, __y)

# < ###########################################################################
