"""Setup the hyper parameters for tokun."""

def rates(pretrained: bool=False, normalization: bool=True, base: float=0.001) -> tuple:
    return (
        (0.1 if pretrained else 1.) * (0.1 if normalization else 1.) * base, # lr max
        0.1, # lambda => global decay
        0.9, # beta_1 => decay for the first moment
        0.99) # beta_2 => decay for the second moment

def version(token_units: list, sequence_axis: int=1, input_dim: int=256, output_dim: int=256) -> list:
    return ['{}x{}'.format(input_dim, output_dim), 'x'.join(str(__u) for __u in token_units), str(sequence_axis)]
