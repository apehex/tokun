# tokun

<img src="header.png" alt="Neural tokenization" title="Source: Image by Author and generated with MidJourney" width="100%" style="margin: auto;"/>

> **!** this project is largely obsolete, replaced by the layer [TokunEmbedding](https://github.com/apehex/mlable?tab=readme-ov-file#TokunEmbedding)

The patching technique used in image / video model can be used on text as explained in [this article](https://huggingface.co/blog/apehex/this-title-is-already-tokenized).

In short, this method reduces 2D spatial data into a 1D sequence fit for transformer architectures.

Conversely, text data can be treated as 2D as follows:

- a scalar tensor of `B` strings is encoded using UTF-32-BE: `(B,) => (B, 4S)`
- the bytes are grouped by chunks of `N`: `(B, 4S) => (B, 4S/N, N)`
- the bytes are embeded independently: `(B, 4S/N, N) => (B, 4S/N, N, E)`
- the embeddings are merged N by N: `(B, 4S/N, N, E) => (B, 4S/N, NE)`

`S` is the limit length for the string inputs and the factor 4 is the number of bytes per character.

The merged byte embeddings form actual "token" embeddings, while keeping the information on composition.
Hence the name "composite emheddings".

There is no more need for a VAE or any model to learn token or sentence embeddings.

## Overview

> `to-kun` took tokens to t-can

Current tokenizers have notorious issues that are bringing all the LLMs down.

`tokun` is a model specialized in text embedding.
It is **lossless** while providing **high input compression**.

`tokun` produces vectors of dimension 256 equivalent to 64 UTF-32-BE bytes.
IE each embedding can be thought of as a *token of length 16 characters*.

But these vectors are more than basic IDs, they keep meaningful information on their constituting parts.

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

In all cases, the model requires the code from the package `tokun`:

```shell
pip install tokun
```

### From Hugging Face

Login to Hugging Face:

```shell
huggingface-cli login
```

Download the repository:

```python
import huggingface_hub as hh

api = hh.HfApi()
api.snapshot_download(repo_id='apehex/tokun', local_dir='tokun/')
```

Import the tokenizer and model:

```python
tokenizer = tokun.huggingface.ByteTokenizer()
model = hh.from_pretrained_keras('tokun/variants/16x4/')
```

### With Base Tensorflow / Keras

You can directly load the weights [from the repository](../models/).

For the most performant variant of the model, `16x4`:

```python
import tensorflow as tf
import tokun.model
import urllib.request

urllib.request.urlretrieve('https://github.com/apehex/tokun/raw/main/models/16x4/1/7.7.keras', 'model.keras')
model = tf.keras.models.load_model('model.keras', compile=False)
```

## Usage

Since it is small (between 1 and 2M parameters depending on the variant), the model can also be [trained on Google Colab][notebook-file-tokun-train].

We will be encoding and decoding the following sample:

```python
__s = """Une unité lexicale ou token lexical ou plus simplement token est un couple composé d'un nom et d'une valeur optionnelle (e.g. 135677)."""
```

### With Hugging Face

The sequence dimension is fixed to 512 because exporting the Keras model requires to specify the input shape.
So the sample is padded to `16 * 512` characters or `64 * 512` bytes.

```python
# encode with UTF-32
__x = tokenizer.batch_encode_plus(batch_text_or_text_pairs=[__s], padding='max_length', max_length=64 * 512, add_special_tokens=False)
__x = tf.convert_to_tensor(__x['input_ids'])
# tokenize
__e = model.layers[1](__x) # encoder
# these embeddings would be the input of a LLM
__o = llm(__e) # replace with your LLM
# detokenize
__p = model.layers[2](__o) # decoder
# interpret probabilities as byte indexes
__y = tokun.pipeline.postprocess(__p)
```

```python
print(len(__s))
# 252
print(__x.shape) # 16 * 512 characters = 64 * 512 bytes
# (1, 32768)
print(__e.shape) # 512 embeddings
# (1, 512, 256)
print(__p.shape) # back to x shape
# (1, 32768, 256)
```

> Note: the base Tensorflow implementation operates on any sequence dimension (see below)

### With Base Tensorflow / Keras

```python
__x = tokun.pipeline.preprocess(text=__s, groups=[4, 16], expand=[1], flatten=True)
__e = model._encoder(__x) # final embedding = input for another model
# these embeddings would be the input of a LLM
__o = llm(__e) # replace with your LLM
# detokenize
__p = MODEL._decoder(__o)
# interpret probabilities as byte indexes
__y = tokun.pipeline.postprocess(__p)
```

The OG version doesn't fix the sequence dimension:

```python
print(len(__s))
# 252
print(__x.shape) # 4 * 252 = 1008 padded to 1024 bytes
# (1, 1024)
print(__e.shape) # 252 / 16 = 1024 / 64 = 16
# (1, 16, 256)
print(__p.shape) # back to x shape
# (1, 1024, 256)
```

## Training and evaluation data

`tokun` was **trained on random sequences** of UTF-32-BE bytes, so that it covers the first 4 planes of Unicode.

Validation was also performed on the 7 languages of [MLQA][github-mlqa] to make sure the model keeps its accuracy on regular text.

## Resources

### Notebooks

Final model:

- train: [file][notebook-file-tokun-train] / [Colab][notebook-colab-tokun-train]
- demo: [file][notebook-file-tokun-demo] / [Colab][notebook-colab-tokun-demo]

Older / simpler model iterations:

- `tokun-1`: [file][notebook-file-tokun-1] / [Colab][notebook-colab-tokun-1]
- `tokun-4`: [file][notebook-file-tokun-4] / [Colab][notebook-colab-tokun-4]
- `tokun-16`: [file][notebook-file-tokun-16] / [Colab][notebook-colab-tokun-16]

### Articles

Main article:

- on [Github][article-file-tokun]
- on [Hugging Face][article-hugging-face]

Notes on each iteration:

- `tokun-1`: [Github][article-file-tokun-1]
- `tokun-4`: [Github][article-file-tokun-4]
- `tokun-16`: [Github][article-file-tokun-16]

## TODO

See [TODO](TODO.md).

## Credits

This project was inspired by a video from Andrej Karpathy, ["Let's build the GPT tokenizer"][youtube-karpathy-tokenizer].

## License

Licensed under the [aGPLv3](LICENSE.md).

[article-file-tokun]: ../articles/tokun.md
[article-file-tokun-1]: ../articles/tokun.1.md
[article-file-tokun-4]: ../articles/tokun.4.md
[article-file-tokun-16]: ../articles/tokun.16.md
[article-hugging-face]: https://huggingface.co/blog/apehex/tokenization-is-a-dead-weight
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
