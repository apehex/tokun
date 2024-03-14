"""Personal take on the tutorial by Andrej Karpathy: https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/"""

import functools
import random

import torch

import mlable.tokens.ngrams as _mtn

# META ########################################################################

N_VOCABULARY = 37
N_CONTEXT = 8
N_EMBEDDING = 32
N_HIDDEN = 256
N_SAMPLE = 32

N_SEED = 42
N_EPOCHS = 4
N_STEPS = 1024
N_BATCH = 4096

R_MIN = 0.01
R_MAX = 0.1
R_EXP = .8

VERSION = 'cnn-torch-300k'

# DATA ########################################################################

USERNAMES = open('.data/usernames.txt', 'r').read().splitlines()

# filter non-ascii characters
USERNAMES = [__w for __w in USERNAMES if all([ord(__c) < 128 for __c in __w])]

# randomize the order
random.shuffle(USERNAMES)

# VOCABULARY ##################################################################

VOCABULARY = _mtn.vocabulary(''.join(USERNAMES))
N_VOCABULARY = len(VOCABULARY)

# MAPPINGS ####################################################################

MAPPINGS = _mtn.mappings(voc=VOCABULARY)

_stoi = MAPPINGS['encode']
_itos = MAPPINGS['decode']

# DATASETS ####################################################################

def build_dataset(words: list, context: int=N_CONTEXT, depth: int=N_VOCABULARY) -> tuple:
    __x = [_mtn.encode(text=__n, stoi=_stoi) for __w in words for __n in _mtn.context(text=__w + _mtn.BLANK, length=context)]
    __y = [__i for __w in words for __i in _mtn.encode(text=__w + _mtn.BLANK, stoi=_stoi)]
    return torch.tensor(__x), torch.tensor(__y)

N1 = int(0.8 * len(USERNAMES))
N2 = int(0.9 * len(USERNAMES))

X_TRAIN, Y_TRAIN = build_dataset(words=USERNAMES[:N1], context=N_CONTEXT)
X_DEV, Y_DEV = build_dataset(words=USERNAMES[N1:N2], context=N_CONTEXT)
X_TEST, Y_TEST = build_dataset(words=USERNAMES[N2:], context=N_CONTEXT)

# LAYERS ######################################################################

class Linear:
    def __init__(self, in_features: int, out_features: int, bias: bool=True) -> None:
        self._weight = torch.randn((in_features, out_features)) / (in_features ** 0.5)
        self._bias = torch.zeros(out_features, requires_grad=True) if bias else None
        # the calculation above returns a new tensor, so setting the flag on creation doesn't work
        self._weight.requires_grad = True

    def __call__(self, x: torch.Tensor, **kwargs: dict) -> torch.Tensor:
        self.out = torch.matmul(x, self._weight)
        if self._bias is not None:
            self.out += self._bias
        return self.out

    def parameters(self) -> list:
        return [self._weight] + ([] if self._bias is None else [self._bias])

class BatchNorm1d:
    def __init__(self, dim: int, epsilon: float=1e-5, momentum: float=0.1) -> None:
        self._epsilon = epsilon
        self._momentum = momentum
        # parameters (trained with backprop)
        self._gamma = torch.ones(dim, requires_grad=True)
        self._beta = torch.zeros(dim, requires_grad=True)
        # buffers (trained with a running 'momentum update')
        self._mean = torch.zeros(dim)
        self._var = torch.ones(dim)
  
    def __call__(self, x: torch.Tensor, training: bool, **kwargs: dict) -> torch.Tensor:
        # current mean
        if training:
            __axes = list(range(x.ndim - 1)) # reduce all axes except the last one
            with torch.no_grad():
                __mean = x.mean(__axes, keepdim=True) # batch mean
                __var = x.var(__axes, keepdim=True) # batch variance
                self._mean = (1. - self._momentum) * self._mean + self._momentum * __mean
                self._var = (1. - self._momentum) * self._var + self._momentum * __var
        # normalize x
        __x = (x - self._mean) / torch.sqrt(self._var + self._epsilon)
        # scale
        self.out = self._gamma * __x + self._beta
        return self.out
  
    def parameters(self) -> list:
        return [self._gamma, self._beta]

class Tanh:
    def __call__(self, x: torch.Tensor, **kwargs: dict) -> torch.Tensor:
        self.out = torch.tanh(x)
        return self.out

    def parameters(self) -> list:
        return []

class Embedding:
    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        self._depth = num_embeddings
        self._weight = torch.randn((num_embeddings, embedding_dim), requires_grad=True)

    def __call__(self, x: torch.Tensor, **kwargs: dict) -> torch.Tensor:
        __x = torch.nn.functional.one_hot(input=x, num_classes=self._depth)
        self.out = torch.matmul(__x.float(), self._weight)
        return self.out

    def parameters(self) -> list:
        return [self._weight]

class Merge:
    def __init__(self, n: int, axis: int) -> None:
        self._n = n
        self._axis = axis

    def __call__(self, x: torch.Tensor, **kwargs: dict) -> torch.Tensor:
        __shape = list(x.shape)
        __axis0 = self._axis % len(__shape)
        __axis1 = (self._axis + 1) % len(__shape)
        # merge n rows along the given axis
        __shape[__axis0] = __shape[__axis0] // self._n
        __shape[__axis1] = __shape[__axis1] * self._n
        # reshape
        self.out = x.view(*__shape).squeeze(1)
        return self.out

    def parameters(self) -> list:
        return []

# BLOCK #######################################################################

class Sequential:
    def __init__(self, layers: list) -> None:
        self._layers = layers

    def __call__(self, x: torch.Tensor, training: bool=True, **kwargs) -> torch.Tensor:
        self.out = x
        # forward
        for __l in self._layers:
            self.out = __l(x=self.out, training=training, **kwargs)
        # conclude
        return self.out

    def parameters(self) -> list:
        return [__p for __l in self._layers for __p in __l.parameters()]

# TRAIN #######################################################################

def batch(x: torch.Tensor, y: torch.Tensor, size: int=N_BATCH) -> tuple:
    __indices = torch.randint(0, x.shape[0], (size,))
    return x[__indices], y[__indices]

def rate(epoch: int, lr_mtn: float, lr_max: float, lr_exp: float, rampup: int, sustain: int) -> float:
    __lr = lr_mtn
    if epoch < rampup:
        __lr = lr_mtn + (epoch * (lr_max - lr_mtn) / rampup)
    elif epoch < rampup + sustain:
        __lr = lr_max
    else:
        __lr = lr_mtn + (lr_max - lr_mtn) * lr_exp ** (epoch - rampup - sustain)
    return __lr

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

# EVALUATE ####################################################################

@torch.no_grad()
def evaluate(model: Sequential, loss: callable, x: torch.Tensor, y: torch.Tensor) -> float:
    __logits = model(x=x, training=True)
    __loss = loss(input=__logits, target=y)
    return __loss.item()

# SAMPLE ######################################################################

def _next(model: Sequential, ngram: list) -> int:
    __logits = model(torch.tensor([ngram]), training=False)
    __probs = torch.nn.functional.softmax(__logits, dim=-1)
    return torch.multinomial(__probs, num_samples=1).item()

def sample(model: Sequential, context: int, length: int, itos: callable) -> str:
    __result = ''
    __ngram = context * [0]
    for __i in range(length):
        __n = _next(model=model, ngram=__ngram)
        __result += itos(__n)
        __ngram = __ngram[1:] + [__n]
    return __result.split('\x00')[0]

# MODEL #######################################################################

torch.manual_seed(N_SEED)

MODEL = Sequential(layers=[
    Embedding(num_embeddings=N_VOCABULARY, embedding_dim=N_EMBEDDING),
    Merge(n=2, axis=-2), Linear(in_features=N_EMBEDDING * 2, out_features=N_HIDDEN, bias=False), BatchNorm1d(dim=N_HIDDEN), Tanh(),
    Merge(n=2, axis=-2), Linear(in_features=N_HIDDEN * 2, out_features=N_HIDDEN, bias=False), BatchNorm1d(dim=N_HIDDEN), Tanh(),
    Merge(n=2, axis=-2), Linear(in_features=N_HIDDEN * 2, out_features=N_HIDDEN, bias=False), BatchNorm1d(dim=N_HIDDEN), Tanh(),
    Linear(in_features=N_HIDDEN, out_features=N_VOCABULARY)])

# MAIN ########################################################################

print(sum(__p.nelement() for __p in MODEL.parameters()))

sgd(model=MODEL, x=X_TRAIN, y=Y_TRAIN, n_epoch=N_EPOCHS, n_batch=N_BATCH)

# print('train loss: {loss}'.format(loss=evaluate(model=MODEL, loss=torch.nn.functional.cross_entropy, x=X_TRAIN, y=Y_TRAIN)))
# print('development loss: {loss}'.format(loss=evaluate(model=MODEL, loss=torch.nn.functional.cross_entropy, x=X_DEV, y=Y_DEV)))
print('validation loss: {loss}'.format(loss=evaluate(model=MODEL, loss=torch.nn.functional.cross_entropy, x=X_TEST, y=Y_TEST)))

_sample = functools.partial(sample, model=MODEL, context=N_CONTEXT, length=128, itos=_itos)
