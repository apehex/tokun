# tokun

> `to-kun` took tokens to t-can

Current tokenizers have notorious issues that are bringing all the LLMs down.
For example I could not get ChatGPT to produce a decent catch-phrase (so you're stuck with mine!).

`tokun` is a NN model specialized in text tokenization.
It produces 256-embedding vectors with a 1:1 match to 64 UTF-32 bytes.

IE each `tokun` embedding can be thought of as a token of length 16 characters.
But these vectors keep meaningful information on their constituting parts.

## Installation

### Using HuggingFace

### From The Weights

## Usage

### Tokenization

#### External

#### Internal

### Fine-Tuning

## Resources

### Models

The main variant of the model is `tokun-16`.

Its hyper-parameters are:

```python
ATTENTION = True # whether the blocks include an attention layer
NORMALIZATION = True # whether the blocks include a normalization layer

N_DEPTH = 3 # D, the number of successive token groupings
N_TOKEN_DIM = 4 # G, the size of a single token group, also the factor of compression
N_ENCODING_DIM = 256 # U, then dimension of a single byte as a one-hot vector
N_EMBEDDING_DIM = N_ENCODING_DIM # E, the dimension of each group
N_LATENT_DIM = N_EMBEDDING_DIM # L, the dimension of the resulting embedding
```

### Notebooks

- `tokun-1`: [File][notebook-file-tokun-1] / [Colab][notebook-colab-tokun-1] / [Kaggle][notebook-kaggle-tokun-1] / [Hugging Face][notebook-hf-tokun-1]
- `tokun-4`: [File][notebook-file-tokun-4] / [Colab][notebook-colab-tokun-4] / [Kaggle][notebook-kaggle-tokun-4] / [Hugging Face][notebook-hf-tokun-4]
- `tokun-16`: [File][notebook-file-tokun-16] / [Colab][notebook-colab-tokun-16] / [Kaggle][notebook-kaggle-tokun-16] / [Hugging Face][notebook-hf-tokun-16]

### Articles

- `tokun-1`: [File][article-file-tokun-1] / [Notion][article-notion-tokun-1] / [Medium][article-medium-tokun-1]
- `tokun-4`: [File][article-file-tokun-4] / [Notion][article-notion-tokun-4] / [Medium][article-medium-tokun-4]
- `tokun-16`: [File][article-file-tokun-16] / [Notion][article-notion-tokun-16] / [Medium][article-medium-tokun-16]

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
[notebook-file-tokun-1]: ../notebooks/tokun.1.ipynb
[notebook-file-tokun-4]: ../notebooks/tokun.4.ipynb
[notebook-file-tokun-16]: ../notebooks/tokun.16.ipynb
[notebook-hf-tokun-1]: ../notebooks/tokun.1.ipynb
[notebook-hf-tokun-4]: ../notebooks/tokun.4.ipynb
[notebook-hf-tokun-16]: ../notebooks/tokun.16.ipynb
[notebook-kaggle-tokun-1]: ../notebooks/tokun.1.ipynb
[notebook-kaggle-tokun-4]: ../notebooks/tokun.4.ipynb
[notebook-kaggle-tokun-16]: ../notebooks/tokun.16.ipynb

[youtube-karpathy-tokenizer]: https://www.youtube.com/watch?v=zduSFxRajkE
