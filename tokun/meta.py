def rates(normalization: bool=True) -> tuple:
    return (0.00001, 0.0001, 0.8) if normalization else (0.0001, 0.001, 0.8)

def version(depth: int=3, unit: int=4, attention: bool=True, normalization: bool=True, framework: str='') -> list:
    __meta = []
    __meta.append(str(unit ** (depth - 1)))
    if framework: __meta.append(framework)
    if attention: __meta.append('attention')
    if normalization: __meta.append('normalization')
    return __meta
