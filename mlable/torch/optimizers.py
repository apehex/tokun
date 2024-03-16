import functools

import torch

import mlable.torch.data as _mtd

# LEARNING RATE ###############################################################

def learning_rate_waveform(step: int, lr_min: float, lr_max: float, lr_exp: float, rampup: int, sustain: int, steps_per_epoch: int=1024) -> float:
    __lr = lr_min
    __epoch = step // steps_per_epoch
    if __epoch < rampup:
        __lr = lr_min + (__epoch * (lr_max - lr_min) / rampup)
    elif __epoch < rampup + sustain:
        __lr = lr_max
    else:
        __lr = lr_min + (lr_max - lr_min) * lr_exp ** (__epoch - rampup - sustain)
    return __lr

# SGD #########################################################################

class SGD(torch.optim.Optimizer):
    def __init__(self, params: list, rate: callable, **kwargs) -> None:
        __default_rate = functools.partial(learning_rate_waveform, lr_min=0.00001, lr_max=0.0001, lr_exp=0.8, rampup=4, sustain=2, steps_per_epoch=1024)
        super(SGD, self).__init__(params, {'rate': __default_rate}, **kwargs)
        self._parameters = list(params)
        self._rate = rate
        self._iteration = -1

    def step(self) -> None:
        self._iteration += 1
        with torch.no_grad():
            for __p in self._parameters:
                __p += -self._rate(self._iteration) * __p.grad

# GENERIC #####################################################################

def step(model: torch.nn.Module, loss: callable, optimizer: torch.optim.Optimizer, x: torch.Tensor, y: torch.Tensor, epoch: int) -> torch.Tensor:
    # forward
    __output = model(x=x, training=True)
    __loss = loss(input=__output, target=y)
    # backward
    model.zero_grad(set_to_none=True)
    __loss.backward()
    # update the parameters
    optimizer.step()
    return __loss

def train(model:torch.nn.Module, loss: callable, optimizer: torch.optim.Optimizer, x: torch.Tensor, y: torch.Tensor, n_epoch: int, n_batch: int) -> None:
    # scheme
    __steps = int(x.shape[0]) // n_batch
    # iterate on the whole dataset
    for __e in range(n_epoch):
        # iterate on batchs
        for __s in range(__steps):
            # track the overall iteration
            __k = __e * __steps + __s
            # random batch
            __x, __y = _mtd.batch(x=x, y=y, size=n_batch)
            # step
            __loss = step(model=model, loss=loss, optimizer=optimizer, x=__x, y=__y, epoch=__e)
            # log the progress
            if __s % __steps == 0:
                print('[epoch {epoch}] train loss: {train}'.format(epoch=__e, train=__loss.item()))
