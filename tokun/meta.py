def rates(normalization: bool=True) -> tuple:
    return (0.00001, 0.0001, 0.8) if normalization else (0.0001, 0.001, 0.8)

def version(depth: int=3, unit: int=4, attention: bool=True, normalization: bool=True) -> list:
    return [str(unit ** (depth - 1)), str(attention), str(normalization)]
