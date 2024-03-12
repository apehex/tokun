import torch

import mlable.torch.datasets as _mtd

# LEARNING RATE ###############################################################

def rate(epoch: int, lr_mtn: float, lr_max: float, lr_exp: float, rampup: int, sustain: int) -> float:
    __lr = lr_mtn
    if epoch < rampup:
        __lr = lr_mtn + (epoch * (lr_max - lr_mtn) / rampup)
    elif epoch < rampup + sustain:
        __lr = lr_max
    else:
        __lr = lr_mtn + (lr_max - lr_mtn) * lr_exp ** (epoch - rampup - sustain)
    return __lr

# SGD #########################################################################

def step(model: Sequential, loss: callable, x: torch.Tensor, y: torch.Tensor, lr: float=0.001) -> torch.Tensor:
    # forward
    __logits = model(x=x, training=True)
    __loss = loss(input=__logits, target=y)
    # backward
    for __p in model.parameters(): __p.grad = None
    __loss.backward()
    # SGD update
    with torch.no_grad():
        for __p in model.parameters():
            __p += -lr * __p.grad
    return __loss

def sgd(model:Sequential, x: torch.Tensor, y: torch.Tensor, n_epoch: int=N_EPOCHS, n_batch: int=N_BATCH) -> None:
    # scheme
    __steps = int(x.shape[0]) // n_batch
    # iterate on the whole dataset
    for __e in range(n_epoch):
        # learning rate
        __lr = rate(epoch=__e, lr_mtn=R_MIN, lr_max=R_MAX, lr_exp=R_EXP, rampup=4, sustain=0)
        # iterate on batchs
        for __s in range(__steps):
            # track the overall iteration
            __k = __e * __steps + __s
            # random batch
            __x, __y = batch(x=x, y=y, size=n_batch)
            # step
            __loss = step(model=model, loss=torch.nn.functional.cross_entropy, x=__x, y=__y, lr=__lr)
            # log the progress
            if __s % __steps == 0:
                print('[epoch {epoch}] train loss: {train}'.format(epoch=__e, train=__loss.item()))
