import torch

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
