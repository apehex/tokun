# CONSTANTS ###################################################################

BLANK = chr(0)

# LIST ########################################################################

def capture(text: str, blank: str=BLANK) -> str:
    return sorted(list(set(text).union({blank})))

# MAPPINGS ####################################################################

def mappings(vocabulary: list) -> dict:
    __itos = {__i: __c for __i, __c in enumerate(vocabulary)}
    __stoi = {__c: __i for __i, __c in enumerate(vocabulary)}
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
