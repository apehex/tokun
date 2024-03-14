import torch

# BATCH #######################################################################

def batch(x: torch.Tensor, y: torch.Tensor, size: int) -> tuple:
    __indices = torch.randint(0, x.shape[0], (size,))
    return x[__indices], y[__indices]
