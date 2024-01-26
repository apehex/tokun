N_CONTEXT = 8

# TEXT TO VECTOR ##############################################################

def tokenize(text: str, length: int=N_CONTEXT, blank='</>'):
    __context = length * blank
    for __c in text + blank:
        yield __context
        __context = __context[1:] + __c
