# CONSTANTS ###################################################################

N_CONTEXT = 8
BLANK = '$'

# TEXT TO VECTOR ##############################################################

def tokenize(text: str, length: int=N_CONTEXT, blank=BLANK):
    __context = length * blank
    for __c in text:
        yield __context
        __context = __context[1:] + __c
