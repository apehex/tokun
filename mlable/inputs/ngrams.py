import mlable.inputs.vocabulary as _miv

# TEXT TO LIST ################################################################

def context(text: str, length: int, blank=_miv.BLANK):
    __context = length * blank
    for __c in text:
        yield __context
        __context = __context[1:] + __c

# TEXT TO VECTOR ##############################################################

def tokenize(text: list, stoi: callable, context_length: int) -> tuple:
    __x = [_miv.encode(text=__n, stoi=stoi) for __n in context(text=text, length=context_length)]
    __y = _miv.encode(text=text, stoi=stoi)
    return (__x, __y)
