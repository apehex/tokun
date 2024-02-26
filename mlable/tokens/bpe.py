import collections
import itertools

# ENCODING ####################################################################

def encode(text: str) -> list:
    return list(text.encode('utf-8'))

def decode(codes: list) -> str:
    return bytes(codes).decode(encoding='utf-8', errors='replace')

# PREPROCESS ##################################################################

def preprocess(codes: list) -> list:
    return [(__i,) if isinstance(__i, int) else __i for __i in codes] # data already preprocessed is left unchanged

# STATS #######################################################################

def count(tokens: list) -> dict:
    return collections.Counter(zip(tokens, tokens[1:]))

# TRAIN #######################################################################

def combine(tokens: list) -> tuple: # tuples are hashable and can be used as keys in dict
    return tuple(itertools.chain.from_iterable(tokens)) # works even on a list with a single element like [(115, 32)] => (115, 32)

def replace(tokens: list, token: tuple) -> list:
    __i = 0
    __r = []
    while __i < len(tokens):
        __p = combine(tokens[__i:__i + 2]) # slice works even when __i = len - 1
        if __p == token:
            __r.append(__p)
            __i += 2
        else:
            __r.append(tokens[__i])
            __i += 1
    return __r

def step(tokens: list) -> list:
    # find the most frequent pair
    __counts = count(tokens)
    __pair = max(__counts, key=__counts.get)
    # combine the two tokens into a single one
    __token = combine(__pair)
    # swap the pair with the new token
    return replace(tokens=tokens, token=__token), __token

def train(tokens: list, steps: int) -> list:
    # init
    __current = tokens
    __pairings = []
    # iterate
    for _ in range(steps):
        __current, __token = step(tokens=__current)
        __pairings.append(__token)
    # return
    return __current, __pairings

# MAIN ########################################################################

def tokenize(codes: list, pairings: list) -> list:
    # init with single tokens
    __current = preprocess(codes=codes)
    # replace the tokens in the same order as the training combinations
    for __t in pairings:
        __current = replace(tokens=__current, token=__t)
    return __current

def detokenize(tokens: list) -> list:
    return list(combine(tokens=tokens))
