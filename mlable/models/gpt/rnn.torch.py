"""RNN following https://www.fit.vutbr.cz/research/groups/speech/publi/2010/mikolov_interspeech2010_IS100722.pdf"""

import functools
import os
import datetime

import torch

import mlable.tokens.ngrams as _mtn
import mlable.torch.layers as _mtl
import mlable.torch.optimizers as _mto
import mlable.torch.sampling as _mts

# META ########################################################################

N_VOCABULARY = 37
N_CONTEXT = 16
N_EMBEDDING = 64
N_BLOCKS = 2
N_HEADS = 4
N_STATE = 64

N_EPOCHS = 2
N_BATCH = 128

N_SAMPLE = 256

R_MIN = 0.0001
R_MAX = 0.001
R_EXP = .8
R_DECAY = 0.01

# IO ##########################################################################

VERSION = 'rnn-torch-180k'

LOGS_PATH = os.path.join('.logs/', VERSION, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

LOAD_PATH = os.path.join('.models/', VERSION, '_', 'model.pt')
SAVE_PATH = os.path.join('.models/', VERSION, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), 'model.pt')

os.makedirs(LOGS_PATH, exist_ok=True)
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

# WRITER = torch.utils.tensorboard.SummaryWriter(log_dir=LOGS_PATH)

# DATA ########################################################################

TEXT = open('.data/shakespeare/othello.md', 'r').read() # .splitlines()
TEXT += open('.data/shakespeare/hamlet.md', 'r').read() # .splitlines()

# VOCABULARY ##################################################################

VOCABULARY = _mtn.vocabulary(TEXT)
N_VOCABULARY = len(VOCABULARY)

# MAPPINGS ####################################################################

MAPPINGS = _mtn.mappings(voc=VOCABULARY)

_stoi = MAPPINGS['encode']
_itos = MAPPINGS['decode']

# DATASET #####################################################################

N1 = int(0.8 * len(TEXT))
N2 = int(0.9 * len(TEXT))

__x, __y = _mtn.tokenize(text=TEXT, stoi=_stoi, context_length=N_CONTEXT)
__X, __Y = torch.Tensor(__x).type(dtype=torch.int64), torch.nn.functional.one_hot(input=torch.Tensor(__y).type(dtype=torch.int64), num_classes=N_VOCABULARY).type(torch.float32)

X_TRAIN, Y_TRAIN = __X[:N1], __Y[:N1]
X_DEV, Y_DEV = __X[N1:N2], __Y[N1:N2]
X_TEST, Y_TEST = __X[N2:], __Y[N2:]

# RNN MODEL ###################################################################

class RNN(torch.nn.Module):
    def __init__(self, time_dim: int, token_dim: int, embed_dim: int, state_dim: int, **kwargs) -> None:
        super(RNN, self).__init__(**kwargs)
        self._state = torch.nn.Parameter(torch.zeros(1, state_dim), requires_grad=True)
        self._embed = _mtl.Embedding(num_embeddings=token_dim, embedding_dim=embed_dim)
        self._cell = _mtl.GRUCell(embed_dim=embed_dim, state_dim=state_dim)
        self._head = torch.nn.Sequential(
            torch.nn.Flatten(start_dim=1, end_dim=2),
            _mtl.Linear(in_features=time_dim * state_dim, out_features=token_dim))

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        __shape = list(x.shape)
        # embedding
        __xt = self._embed(x) # (B, T, E)
        # start from internal state
        __ht = self._state.expand((__shape[0], -1)) # expand out the batch dimension
        __h = []
        # combine with successive inputs in context
        for i in range(__shape[1]):
            __ht = self._cell(__xt[:, i, :] , __ht) # (B, E)
            __h.append(__ht)
        # decode the outputs
        return self._head(torch.stack(__h, 1))

# TRAIN ########################################################################

def main(model: torch.nn.Module, loss: callable=torch.nn.functional.cross_entropy, x: torch.Tensor=X_TRAIN, y: torch.Tensor=Y_TRAIN, n_epoch: int=N_EPOCHS, n_batch: int=N_BATCH, lr: float=R_MAX, decay: float=R_DECAY, training: bool=True) -> None:
    # train
    if training:
        __optimizer = torch.optim.AdamW(model.parameters(recurse=True), lr=lr, weight_decay=decay, betas=(0.9, 0.99), eps=1e-8)
        _mto.train(model=model, loss=loss, optimizer=__optimizer, x=x, y=y, n_epoch=n_epoch, n_batch=n_batch)
    # evaluate

# MAIN ########################################################################

if __name__ == '__main__':
    __model = RNN(time_dim=N_CONTEXT, token_dim=N_VOCABULARY, embed_dim=N_EMBEDDING, state_dim=N_STATE)
    # load pre-trained weights
    if os.path.isfile(LOAD_PATH): __model.load_state_dict(torch.load(LOAD_PATH))
    # train
    main(__model)
    # save for later
    torch.save(__model.state_dict(), SAVE_PATH)
    # generate sample text
    __s = _mts.sample(model=__model, context=N_CONTEXT, length=2**10)
    print(_mtn.decode(sequence=__s, itos=_itos))
