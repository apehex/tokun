"""Personal take on the tutorial by Andrej Karpathy: https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/"""

import random

import torch

import mlable.inputs.ngrams as _min
import mlable.inputs.vocabulary as _miv

# META ########################################################################

N_VOCABULARY = 37
N_CONTEXT = 8
N_EMBEDDING = 32
N_hidden = 256
N_SAMPLE = 32

N_STEPS = 1024
N_BATCH = 4096

R_TRAINING = 0.2

VERSION = 'cnn-torch-80k'

# DATA ########################################################################

USERNAMES = open('.data/usernames.txt', 'r').read().splitlines()

# filter non-ascii characters
USERNAMES = [__w for __w in USERNAMES if all([ord(__c) < 128 for __c in __w])]

# randomize the order
random.shuffle(USERNAMES)

# VOCABULARY ##################################################################

VOCABULARY = _miv.capture(USERNAMES)
N_VOCABULARY = len(VOCABULARY)

# MAPPINGS ####################################################################

MAPPINGS = _miv.mappings(vocabulary=VOCABULARY)

_stoi = MAPPINGS['encode']
_itos = MAPPINGS['decode']

# DATASETS ####################################################################

def build_dataset(words: list, context: int=N_CONTEXT, depth: int=N_VOCABULARY) -> tuple:
    __x = [_miv.encode(text=__n, stoi=_stoi) for __w in words for __n in _min.context(text=__w, length=context)]
    __y = [__i for __w in words for __i in _miv.encode(text=__w + _miv.BLANK, stoi=_stoi)]
    return torch.tensor(__x), torch.tensor(__y)

N1 = int(0.8 * len(USERNAMES))
N2 = int(0.9 * len(USERNAMES))

X_TRAIN, Y_TRAIN = build_dataset(words=USERNAMES[:N1], context=N_CONTEXT)
X_DEV, Y_DEV = build_dataset(words=USERNAMES[N1:N2], context=N_CONTEXT)
X_TEST, Y_TEST = build_dataset(words=USERNAMES[N2:], context=N_CONTEXT)

# LAYERS ######################################################################

class Linear:
  
  def __init__(self, fan_in, fan_out, bias=True):
    self.weight = torch.randn((fan_in, fan_out)) / fan_in**0.5 # note: kaiming init
    self.bias = torch.zeros(fan_out) if bias else None
  
  def __call__(self, x):
    self.out = x @ self.weight
    if self.bias is not None:
      self.out += self.bias
    return self.out
  
  def parameters(self):
    return [self.weight] + ([] if self.bias is None else [self.bias])

class BatchNorm1d:
  
  def __init__(self, dim, eps=1e-5, momentum=0.1):
    self.eps = eps
    self.momentum = momentum
    self.training = True
    # parameters (trained with backprop)
    self.gamma = torch.ones(dim)
    self.beta = torch.zeros(dim)
    # buffers (trained with a running 'momentum update')
    self.running_mean = torch.zeros(dim)
    self.running_var = torch.ones(dim)
  
  def __call__(self, x):
    # calculate the forward pass
    if self.training:
      if x.ndim == 2:
        dim = 0
      elif x.ndim == 3:
        dim = (0,1)
      xmean = x.mean(dim, keepdim=True) # batch mean
      xvar = x.var(dim, keepdim=True) # batch variance
    else:
      xmean = self.running_mean
      xvar = self.running_var
    xhat = (x - xmean) / torch.sqrt(xvar + self.eps) # normalize to unit variance
    self.out = self.gamma * xhat + self.beta
    # update the buffers
    if self.training:
      with torch.no_grad():
        self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
        self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar
    return self.out
  
  def parameters(self):
    return [self.gamma, self.beta]

class Tanh:
  def __call__(self, x):
    self.out = torch.tanh(x)
    return self.out
  def parameters(self):
    return []

class Embedding:
  
  def __init__(self, num_embeddings, embedding_dim):
    self.weight = torch.randn((num_embeddings, embedding_dim))
    
  def __call__(self, IX):
    self.out = self.weight[IX]
    return self.out
  
  def parameters(self):
    return [self.weight]

class FlattenConsecutive:
  
  def __init__(self, n):
    self.n = n
    
  def __call__(self, x):
    B, T, C = x.shape
    x = x.view(B, T//self.n, C*self.n)
    if x.shape[1] == 1:
      x = x.squeeze(1)
    self.out = x
    return self.out
  
  def parameters(self):
    return []

class Sequential:
  
  def __init__(self, layers):
    self.layers = layers
  
  def __call__(self, x):
    for layer in self.layers:
      x = layer(x)
    self.out = x
    return self.out
  
  def parameters(self):
    # get parameters of all layers and stretch them out into one list
    return [p for layer in self.layers for p in layer.parameters()]

torch.manual_seed(42); # seed rng for reproducibility

# MODEL #######################################################################

model = Sequential([
  Embedding(N_VOCABULARY, N_EMBEDDING),
  FlattenConsecutive(2), Linear(N_EMBEDDING * 2, N_hidden, bias=False), BatchNorm1d(N_hidden), Tanh(),
  FlattenConsecutive(2), Linear(N_hidden*2, N_hidden, bias=False), BatchNorm1d(N_hidden), Tanh(),
  FlattenConsecutive(2), Linear(N_hidden*2, N_hidden, bias=False), BatchNorm1d(N_hidden), Tanh(),
  Linear(N_hidden, N_VOCABULARY),
])

# parameter init
with torch.no_grad():
  model.layers[-1].weight *= 0.1 # last layer make less confident

parameters = model.parameters()
print(sum(p.nelement() for p in parameters)) # number of parameters in total
for p in parameters:
  p.requires_grad = True

# same optimization as last time
max_steps = 200000
batch_size = 32
lossi = []

for i in range(max_steps):
  
  # minibatch construct
  ix = torch.randint(0, X_TRAIN.shape[0], (batch_size,))
  Xb, Yb = X_TRAIN[ix], Y_TRAIN[ix] # batch X,Y
  
  # forward pass
  logits = model(Xb)
  loss = torch.nn.functional.cross_entropy(logits, Yb) # loss function
  
  # backward pass
  for p in parameters:
    p.grad = None
  loss.backward()
  
  # update: simple SGD
  lr = 0.1 if i < 150000 else 0.01 # step learning rate decay
  for p in parameters:
    p.data += -lr * p.grad

  # track stats
  if i % 10000 == 0: # print every once in a while
    print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}')
  lossi.append(loss.log10().item())

plt.plot(torch.tensor(lossi).view(-1, 1000).mean(1))

# put layers into eval mode (needed for batchnorm especially)
for layer in model.layers:
  layer.training = False

# evaluate the loss
@torch.no_grad() # this decorator disables gradient tracking inside pytorch
def split_loss(split):
  x,y = {
    'train': (X_TRAIN, Y_TRAIN),
    'val': (X_DEV, Y_DEV),
    'test': (X_TEST, Y_TEST),
  }[split]
  logits = model(x)
  loss = torch.nn.functional.cross_entropy(logits, y)
  print(split, loss.item())

split_loss('train')
split_loss('val')


# sample from the model
for _ in range(20):
    
    out = []
    context = [0] * N_CONTEXT # initialize with all ...
    while True:
      # forward pass the neural net
      logits = model(torch.tensor([context]))
      probs = torch.nn.functional.softmax(logits, dim=1)
      # sample from the distribution
      ix = torch.multinomial(probs, num_samples=1).item()
      # shift the context window and track the samples
      context = context[1:] + [ix]
      out.append(ix)
      # if we sample the special '.' token, break
      if ix == 0:
        break
    
    print(''.join(_itos(i) for i in out)) # decode and print the generated word


for x,y in zip(X_TRAIN[7:15], Y_TRAIN[7:15]):
  print(''.join(_itos(ix.item()) for ix in x), '-->', _itos(y.item()))

# forward a single example:
logits = model(X_TRAIN[[7]])
logits.shape

# forward all of them
logits = torch.zeros(8, 27)
for i in range(8):
  logits[i] = model(X_TRAIN[[7+i]])
logits.shape