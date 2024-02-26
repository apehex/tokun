# CONSTANTS ###################################################################

BLANK = chr(0)

# LIST ########################################################################

def vocabulary(text: str, blank: str=BLANK) -> str:
    return sorted(list(set(text).union({blank})))

# MAPPINGS ####################################################################

def mappings(voc: list) -> dict:
    __itos = {__i: __c for __i, __c in enumerate(voc)}
    __stoi = {__c: __i for __i, __c in enumerate(voc)}
    # blank placeholder
    __blank_c = __itos[0] # chr(0)
    __blank_i = 0
    # s => i
    def __encode(c: str) -> int:
        return __stoi.get(c, __blank_i)
    # i => s
    def __decode(i: int) -> str:
        return __itos.get(i, __blank_c)
    # return both
    return {'encode': __encode, 'decode': __decode}

# ENCODING ####################################################################

def encode(text: str, stoi: callable) -> list:
    return [stoi(__c) for __c in text] # defaults to 0 if a character is not in the vocabulary

def decode(sequence: list, itos: callable) -> list:
    return ''.join([itos(__i) for __i in sequence]) # defaults to the first character

# TEXT TO LIST ################################################################

def context(text: str, length: int, blank=BLANK):
    __context = length * blank
    for __c in text:
        yield __context
        __context = __context[1:] + __c

# TEXT TO VECTOR ##############################################################

def tokenize(text: list, stoi: callable, context_length: int) -> tuple:
    __x = [encode(text=__n, stoi=stoi) for __n in context(text=text, length=context_length)]
    __y = encode(text=text, stoi=stoi)
    return (__x, __y)
