# tokun

> `to-kun` took tokens to t-can

Current tokenizers have notorious issues that are bringing all the LLMs down.
For example I could not get ChatGPT to produce a decent catch-phrase (so you're stuck with mine!).

`tokun` is a NN model specialized in text tokenization.
It produces 256-embedding vectors equivalent to 64 UTF-32-BE bytes.

IE each `tokun` embedding can be thought of as a token of length 16 characters.

But these vectors are more than basic IDs, they keep meaningful information on their constituting parts.

The architecture, dataviz, ambition and results are detailed in the [articles](../articles).

## Features

The model produces vector embeddings that can be directly ingested by another model.

Regular tokens are unrelated IDs, while `tokun` has the following properties:

- **international**: `tokun` performs evenly on the whole Unicode space
- **compression**: the sequence length is divided by 16
- **embeddings**: the output vectors have only a dimension 256
- **lossless**: embeddings store all the information up to the byte level
- **built-ins**: Unicode has built-in special tokens, no need for `<|im_start|>`
- **meaningful**: embeddings are natively related to each-other based on their parts

## Installation

For now, the model is only available here, [on Github](../tokun/).

You can simply clone the repository: in particular, it will download [the weights](../models/).

## Usage

The model can be loaded from the exported weights:

```python
import os
import tensorflow as tf

import tokun.meta # default values for the hyper parameters
import tokun.model # required to register the Keras classes
import tokun.pipeline # pre and post-processing

# META ########################################################################

ATTENTION = True
NORMALIZATION = True

N_TOKEN_DIM = [4, 4] # G, for each block

# DERIVED #####################################################################

LABEL = '8.5'
VERSION = tokun.meta.version(groups=N_TOKEN_DIM, attention=ATTENTION, normalization=NORMALIZATION)

# IMPORT ######################################################################

PATH_IMPORT = os.path.join('models/', *VERSION, '{}.keras'.format(LABEL))

MODEL = tf.keras.models.load_model(PATH_IMPORT)
```

Since it is small (between 1 and 2M parameters depending on the exact configuration), the model can also be [trained on Google Colab][notebook-file-tokun-train].

### Tokenization

Once the model loaded, a text strings `__s` can be encoded with:

```python
__x = tokun.pipeline.preprocess(text=__s, groups=N_TOKEN_DIM, flatten=True)
__e = MODEL._encoder(__x) # final embedding = input for another model
```

### Detokenization

An embedding tensor `__e` (or prediction) can be reversed into Unicode text with:

```python
__p = MODEL._decoder(__e)
__y = tokun.pipeline.postprocess(__p)
```

## Resources

### Models

The most common variations have been trained and exported to the [models subdirectory](../models/).

The main variant of the model is `tokun-16`.

Its hyper-parameters are:

```python
ATTENTION = True # whether the blocks include an attention layer
NORMALIZATION = True # whether the blocks include a normalization layer

N_TOKEN_DIM = [4, 4, 4] # G, the size of the token groups, for each block
N_ENCODING_DIM = 256 # U, then dimension of a single byte as a one-hot vector
N_EMBEDDING_DIM = N_ENCODING_DIM # E, the dimension of each group
N_LATENT_DIM = N_EMBEDDING_DIM # L, the dimension of the resulting embedding
```

### Notebooks

Final model:

- train: [File][notebook-file-tokun-train] / [Colab][notebook-colab-tokun-train]
- demo: [File][notebook-file-tokun-demo] / [Colab][notebook-colab-tokun-demo]

Older / simpler model iterations:

- `tokun-1`: [File][notebook-file-tokun-1] / [Colab][notebook-colab-tokun-1]
- `tokun-4`: [File][notebook-file-tokun-4] / [Colab][notebook-colab-tokun-4]
- `tokun-16`: [File][notebook-file-tokun-16] / [Colab][notebook-colab-tokun-16]

### Articles

- `tokun-1`: [Github][article-file-tokun-1]
- `tokun-4`: [Github][article-file-tokun-4]
- `tokun-16`: [Github][article-file-tokun-16]

## TODO

See [TODO](TODO.md).

## Credits

This project was inspired by a video from Andrej Karpathy, ["Let's build the GPT tokenizer"][youtube-karpathy-tokenizer].

## License

Licensed under the [aGPLv3](LICENSE.md).

[article-file-tokun-1]: ../articles/tokun.1.md
[article-file-tokun-4]: ../articles/tokun.4.md
[article-file-tokun-16]: ../articles/tokun.16.md
[article-medium-tokun-1]: ../articles/tokun.1.md
[article-medium-tokun-4]: ../articles/tokun.4.md
[article-medium-tokun-16]: ../articles/tokun.16.md
[article-notion-tokun-1]: https://apehex.notion.site/Tokun-1-e03c438a39fe49fcb2ce303eb63b2e73
[article-notion-tokun-4]: https://apehex.notion.site/Tokun-4-c8b4a3bd1270485a908287869553e9f2
[article-notion-tokun-16]: https://apehex.notion.site/Tokun-16-ecf35d5207ab401d85d3aa21d0b09538

[notebook-colab-tokun-1]: https://colab.research.google.com/github/apehex/tokun/blob/main/notebooks/tokun.1.ipynb
[notebook-colab-tokun-4]: https://colab.research.google.com/github/apehex/tokun/blob/main/notebooks/tokun.4.ipynb
[notebook-colab-tokun-16]: https://colab.research.google.com/github/apehex/tokun/blob/main/notebooks/tokun.16.ipynb
[notebook-colab-tokun-demo]: https://colab.research.google.com/github/apehex/tokun/blob/main/notebooks/tokun.demo.ipynb
[notebook-colab-tokun-train]: https://colab.research.google.com/github/apehex/tokun/blob/main/notebooks/tokun.train.ipynb
[notebook-file-tokun-1]: ../notebooks/tokun.1.ipynb
[notebook-file-tokun-4]: ../notebooks/tokun.4.ipynb
[notebook-file-tokun-16]: ../notebooks/tokun.16.ipynb
[notebook-file-tokun-demo]: ../notebooks/tokun.demo.ipynb
[notebook-file-tokun-train]: ../notebooks/tokun.train.ipynb
[notebook-hf-tokun-demo]: ../notebooks/tokun.demo.ipynb
[notebook-hf-tokun-train]: ../notebooks/tokun.train.ipynb
[notebook-kaggle-tokun-demo]: ../notebooks/tokun.demo.ipynb
[notebook-kaggle-tokun-train]: ../notebooks/tokun.train.ipynb

[youtube-karpathy-tokenizer]: https://www.youtube.com/watch?v=zduSFxRajkE
