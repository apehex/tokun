import math

import torch

# NORMALIZATION ###############################################################

class BatchNorm1d(torch.nn.Module):
    def __init__(self, dim: int, epsilon: float=1e-5, momentum: float=0.1, **kwargs) -> None:
        super(BatchNorm1d, self).__init__(**kwargs)
        self._epsilon = epsilon
        self._momentum = momentum
        # parameters (trained with backprop)
        self._gamma = torch.nn.Parameter(torch.ones(dim), requires_grad=True)
        self._beta = torch.nn.Parameter(torch.zeros(dim), requires_grad=True)
        # buffers (trained with a running 'momentum update')
        self._mean = torch.zeros(dim)
        self._var = torch.ones(dim)
        self.register_buffer("mean", self._mean)
        self.register_buffer("variance", self._var)
  
    def forward(self, x: torch.Tensor, training: bool, **kwargs) -> torch.Tensor:
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
        return self._gamma * __x + self._beta

# ACTIVATION ##################################################################

class Tanh(torch.nn.Module):
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return torch.tanh(x)

class NewGELU(torch.nn.Module):
    """Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415"""
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

# RESHAPING ###################################################################

class Merge(torch.nn.Module):
    def __init__(self, n: int, axis: int, **kwargs) -> None:
        super(Merge, self).__init__(**kwargs)
        self._n = n
        self._axis = axis

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        __shape = list(x.shape)
        __axis0 = self._axis % len(__shape)
        __axis1 = (self._axis + 1) % len(__shape)
        # merge n rows along the given axis
        __shape[__axis0] = __shape[__axis0] // self._n
        __shape[__axis1] = __shape[__axis1] * self._n
        # reshape
        return x.view(*__shape).squeeze(1)

# LINEAR ######################################################################

class Linear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool=True, **kwargs) -> None:
        super(Linear, self).__init__(**kwargs)
        self._weight = torch.nn.Parameter(torch.randn((in_features, out_features)) / (in_features ** 0.5), requires_grad=True)
        self._bias = torch.nn.Parameter(torch.zeros(out_features), requires_grad=True) if bias else None

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        __x = torch.matmul(x, self._weight)
        if self._bias is not None:
            __x += self._bias
        return __x

class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, **kwargs) -> None:
        super(Embedding, self).__init__(**kwargs)
        self._depth = num_embeddings
        self._weight = torch.nn.Parameter(torch.randn((num_embeddings, embedding_dim)), requires_grad=True)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        __x = torch.nn.functional.one_hot(input=x, num_classes=self._depth)
        return torch.matmul(__x.float(), self._weight)

class PositionalEmbedding(torch.nn.Module):
    def __init__(self, time_dim: int, token_dim: int, embed_dim: int, **kwargs) -> None:
        super(PositionalEmbedding, self).__init__(**kwargs)
        # simultaneous embedding of tokens and position
        self._token_embedding = Embedding(num_embeddings=token_dim, embedding_dim=embed_dim)
        self._position_embedding = Embedding(num_embeddings=time_dim, embedding_dim=embed_dim)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        __shape = list(x.shape)
        # time position
        __p = torch.arange(0, __shape[1], dtype=torch.long).view(1, __shape[1]) # (1, T)
        # combine
        return self._token_embedding(x) + self._position_embedding(__p) # (B, T, E) + (1, T, E)

# RECURRENT ###################################################################

class RNNCell(torch.nn.Module):
    def __init__(self, embed_dim: int, state_dim: int, **kwargs) -> None:
        super(RNNCell, self).__init__(**kwargs)
        self._weights = Linear(in_features=embed_dim + state_dim, out_features=state_dim)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        __xh = torch.cat([x, h], dim=-1)
        return torch.nn.functional.tanh(self._weights(__xh))

class GRUCell(torch.nn.Module):
    def __init__(self, embed_dim: int, state_dim: int, **kwargs) -> None:
        super(GRUCell, self).__init__(**kwargs)
        # input, forget, output, gate
        self._xh_to_z = Linear(in_features=embed_dim + state_dim, out_features=state_dim)
        self._xh_to_r = Linear(in_features=embed_dim + state_dim, out_features=state_dim)
        self._xh_to_hhat = Linear(in_features=embed_dim + state_dim, out_features=state_dim)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        # state
        __xh = torch.cat([x, h], dim=-1)
        # reset gate
        __r = torch.nn.functional.sigmoid(self._xh_to_r(__xh))
        # switch gate
        __z = torch.nn.functional.sigmoid(self._xh_to_z(__xh))
        # reset state
        __xhr = torch.cat([x, __r * h], dim=-1)
        # candidate state
        __hhat = torch.nn.functional.tanh(self._xh_to_hhat(__xhr))
        # combine candidate and previous states
        return (1. - __z) * h + __z * __hhat

# ATTENTION ###################################################################

class CausalSelfAttention(torch.nn.Module):
    def __init__(self, time_dim: int, embed_dim: int, num_heads: int, **kwargs) -> None:
        super(CausalSelfAttention, self).__init__(**kwargs)
        assert embed_dim % num_heads == 0
        # key, query, value projections for all heads, but in a batch
        self._attention = Linear(in_features=embed_dim, out_features=3 * embed_dim)
        # output projection
        self._projection = Linear(in_features=embed_dim, out_features=embed_dim)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self._mask = torch.tril(torch.ones(time_dim, time_dim)).view(1, 1, time_dim, time_dim)
        self.register_buffer("mask", self._mask)
        # save the shape
        self._head_count = num_heads
        self._head_dim = embed_dim

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        # insert a new axis to group by attention head
        __shape = list(x.shape)
        __shape.insert(2, self._head_count)
        __shape[-1] = __shape[-1] // self._head_count
        # calculate query, key, values for all heads in batch
        __q, __k, __v  = self._attention(x).split(self._head_dim, dim=-1)
        # group by head rather than time
        __k = __k.view(*__shape).transpose(1, 2) # (B, H, T, E/H)
        __q = __q.view(*__shape).transpose(1, 2) # (B, H, T, E/H)
        __v = __v.view(*__shape).transpose(1, 2) # (B, H, T, E/H)
        # self-attention
        __w = (__q @ __k.transpose(-2, -1)) * (1.0 / math.sqrt(__shape[-1])) # (B, H, T, E/H) x (B, H, E/H, T) -> (B, H, T, T)
        # causal: only attend to past tokens
        __w = __w.masked_fill(self._mask == 0, float('-inf'))
        __w = torch.nn.functional.softmax(__w, dim=-1)
        # values
        __y = __w @ __v # (B, H, T, T) x (B, H, T, E/H) -> (B, H, T, E/H)
        # assemble heads
        __y = __y.transpose(1, 2).contiguous().view(*x.shape) # original shape (B, T, E)
        # output projection
        return self._projection(__y)

# BLOCKS ######################################################################

class Sequential(torch.nn.Module):
    def __init__(self, layers: list, **kwargs) -> None:
        super(Sequential, self).__init__(**kwargs)
        self._layers = layers

    def forward(self, x: torch.Tensor, training: bool=True, **kwargs) -> torch.Tensor:
        __x = x
        # forward
        for __l in self._layers:
            __x = __l(x=__x, training=training, **kwargs)
        # conclude
        return __x

class TransformerBlock(torch.nn.Module):
    def __init__(self, time_dim: int, embed_dim: int, num_heads: int, **kwargs) -> None:
        super(TransformerBlock, self).__init__(**kwargs)
        self._block = torch.nn.Sequential(
            torch.nn.LayerNorm(embed_dim),
            CausalSelfAttention(time_dim=time_dim, embed_dim=embed_dim, num_heads=num_heads),
            torch.nn.LayerNorm(embed_dim),
            Linear(embed_dim, 4 * embed_dim),
            Linear(4 * embed_dim, embed_dim),
            NewGELU())

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self._block(x, **kwargs)
