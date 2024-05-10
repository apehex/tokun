def rates(normalization: bool=True) -> tuple:
    return (0.00001, 0.0001, 0.8) if normalization else (0.0001, 0.001, 0.8)

def version(groups: list, attention: bool=True, normalization: bool=True) -> list:
    return ['x'.join(str(__g) for __g in groups), str(attention), str(normalization)]
