"""Setup the hyper parameters for tokun."""

def rates(pretrained: bool=False, normalization: bool=True, base: float=0.001) -> tuple:
    return (
        (0.1 if pretrained else 1.) * (0.1 if normalization else 1.) * 0.01 * base, # lr min
        (0.1 if pretrained else 1.) * (0.1 if normalization else 1.) * base, # lr max
        0.8) # lr decay rate

def version(units: list, axis: int=1) -> list:
    return ['x'.join(str(__g) for __g in units), str(axis)]
