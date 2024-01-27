# CONSTANTS ###################################################################

BLANK = '$'

# LIST ########################################################################

def capture(text: str, blank: str=BLANK) -> str:
    return sorted(list(set(text).union({blank})))

# MAPPINGS ####################################################################

def mappings(vocabulary: list, blank=BLANK) -> dict:
    __itos = {__i: __c for __i, __c in enumerate(vocabulary)}
    __stoi = {__c: __i for __i, __c in enumerate(vocabulary)}
    # blank placeholder
    __blank_c = blank
    __blank_i = len(vocabulary)
    # append to vocabulary
    __itos[__blank_i] = __blank_c
    __stoi[__blank_c] = __blank_i
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
    return [stoi(__c) for __c in text]

def decode(sequence: list, itos: callable) -> list:
    return ''.join([itos(__i) for __i in sequence])
