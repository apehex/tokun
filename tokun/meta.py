"""Setup the hyper parameters for tokun."""

def rates(pretrained: bool=False, normalization: bool=True, base: float=0.001) -> tuple:
    return (
        (0.1 if pretrained else 1.) * (0.1 if normalization else 1.) * 0.01 * base, # lr min
        (0.1 if pretrained else 1.) * (0.1 if normalization else 1.) * base, # lr max
        0.8) # lr decay rate

def version(groups: list, activation: str='silu', attention: bool=True, normalization: bool=True) -> list:
    return ['x'.join(str(__g) for __g in groups), str(activation), str(attention), str(normalization)]
