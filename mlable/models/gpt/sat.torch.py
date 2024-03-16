"""Personal take on the self-attention transformer in https://github.com/karpathy/makemore/blob/master/makemore.py"""

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
N_ATTENTION = 64
N_HIDDEN = 4 * N_ATTENTION

N_EPOCHS = 2
N_BATCH = 128

N_SAMPLE = 256

R_MIN = 0.0001
R_MAX = 0.001
R_EXP = .8
R_DECAY = 0.01

# IO ##########################################################################

VERSION = 'sat-torch-180k'

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

# GPT-2 MODEL #################################################################

class Transformer(torch.nn.Module):
    """ Transformer Language Model, exactly as seen in GPT-2 """

    def __init__(self, time_dim: int, token_dim: int, embed_dim: int, block_count: int, head_count: int, **kwargs) -> None:
        super(Transformer, self).__init__(**kwargs)
        # transformer
        self._transformer = torch.nn.Sequential(
            _mtl.PositionalEmbedding(time_dim=token_dim, token_dim=token_dim, embed_dim=embed_dim),
            *[_mtl.TransformerBlock(time_dim=time_dim, embed_dim=embed_dim, num_heads=head_count) for _ in range(block_count)],
            torch.nn.Flatten(start_dim=1, end_dim=2),
            torch.nn.LayerNorm(normalized_shape=time_dim * embed_dim),
            _mtl.Linear(in_features=time_dim * embed_dim, out_features=token_dim, bias=False))

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self._transformer(x)

# TRAIN ########################################################################

def main(model: torch.nn.Module, loss: callable=torch.nn.functional.cross_entropy, x: torch.Tensor=X_TRAIN, y: torch.Tensor=Y_TRAIN, n_epoch: int=N_EPOCHS, n_batch: int=N_BATCH, lr: float=R_MAX, decay: float=R_DECAY, training: bool=True) -> None:
    # train
    if training:
        # SGD optimizer
        # __steps_per_epoch = x.shape[0] // n_batch
        # __rate = functools.partial(_mto.learning_rate_waveform, lr_min=0.05, lr_max=0.1, lr_exp=decay, rampup=1, sustain=0, steps_per_epoch=__steps_per_epoch)
        # __optimizer = _mto.SGD(params=__model.parameters(recurse=True), rate=__rate)
        __optimizer = torch.optim.AdamW(model.parameters(recurse=True), lr=lr, weight_decay=decay, betas=(0.9, 0.99), eps=1e-8)
        _mto.train(model=model, loss=loss, optimizer=__optimizer, x=x, y=y, n_epoch=n_epoch, n_batch=n_batch)
    # evaluate

# MAIN ########################################################################

if __name__ == '__main__':
    __model = Transformer(time_dim=N_CONTEXT, token_dim=N_VOCABULARY, embed_dim=N_ATTENTION, block_count=N_BLOCKS, head_count=N_HEADS)
    # load pre-trained weights
    if os.path.isfile(LOAD_PATH):
        print('[loading] {path}...'.format(path=LOAD_PATH))
        __model.load_state_dict(torch.load(LOAD_PATH))
    # train
    main(__model)
    # save for later
    print('[saving] {path}...'.format(path=SAVE_PATH))
    torch.save(__model.state_dict(), SAVE_PATH)
    # generate sample text
    __s = _mts.sample(model=__model, context=N_CONTEXT, length=2**10)
    print(_mtn.decode(sequence=__s, itos=_itos))
