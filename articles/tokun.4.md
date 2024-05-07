# Tokun-4

> `tokun`

## Resources

Other articles in the serie:

- [`tokun-1`][article-github-tokun-1]
- [`tokun-16`][article-github-tokun-16]

All the variants of the model are already available on:

- [Github][tokun-github]
- [Hugging Face][tokun-huggingface]
- [Kaggle][tokun-kaggle]

You will also find notebooks on:

- [Github][notebook-github]
- [Google Colab][notebook-colab]
- [Hugging Face][notebook-huggingface]
- [Kaggle][notebook-kaggle]

## Summary

The previous model `tokun-1` gave us character level tokens / embeddings that:

1. [x] is an actual neural network
2. [x] generalizes across all languages
3. [x] produces embeddings of dimension 256

So:

Pushing forward:

## Model

### Input & Output

### Architecture

#### Encoder

#### Decoder

## Training

## Results

### Metrics

### Samples

### Generalization Power

#### New Samples

#### New Characters

#### New Languages

## Implementation Details

### Divide Layer

```python
class Divide(tf.keras.layers.Layer):
    def __init__(
        self,
        input_axis: int, # relative to the NEW shape / rank
        output_axis: int, # same
        factor: int,
        insert: bool=False,
        **kwargs
    ) -> None:
        super(Divide, self).__init__(**kwargs)
        self._input_axis = input_axis
        self._output_axis = output_axis
        self._factor = factor
        self._insert = insert

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        # infer the dimension of the symbolic axis
        __shape = _normalize_shape(list(inputs.shape))
        # rank, according to the new shape
        __rank = len(__shape) + int(self._insert)
        # axes, taken from the new shape
        __axis0 = self._input_axis % __rank
        __axis1 = self._output_axis % __rank
        # option to group data on a new axis
        if self._insert: __shape.insert(__axis1, 1)
        # move data from axis 0 to axis 1
        __shape[__axis0] = _divide_dim(__shape[__axis0], self._factor)
        __shape[__axis1] = _multiply_dim(__shape[__axis1], self._factor)
        return tf.reshape(tensor=inputs, shape=__shape)
```

### Merge Layer

```python
class Merge(tf.keras.layers.Layer):
    def __init__(
        self,
        left_axis: int=-2,
        right_axis: int=-1,
        left: bool=True,
        **kwargs
    ) -> None:
        super(Merge, self).__init__(**kwargs)
        self._left_axis = left_axis
        self._right_axis = right_axis
        self._left = left

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        # infer the dimension of the symbolic axis
        __shape = _normalize_shape(list(inputs.shape))
        __rank = len(__shape)
        # target axes
        __axis_l = self._left_axis % __rank
        __axis_r = self._right_axis % __rank
        # new axis
        __dim = _multiply_dim(__shape[__axis_l], __shape[__axis_r])
        __axis_k = __axis_l if self._left else __axis_r # kept axis
        __axis_d = __axis_r if self._left else __axis_l # deleted axis
        # new shape
        __shape[__axis_k] = __dim
        __shape.pop(__axis_d)
        # actually merge the two axes
        return tf.reshape(tensor=inputs, shape=__shape)
```

### Tokenization Block

```python
class TokenizeBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        left_axis: int=-2,
        right_axis: int=-1,
        token_dim: int=4,
        latent_dim: int=256,
        **kwargs
    ) -> None:
        super(TokenizeBlock, self).__init__(**kwargs)
        # layers
        self._divide = _mtl.Divide(input_axis=0, output_axis=1, factor=token_dim, insert=True, name='group') # (B * G, E) => (B, G, E)
        self._embedding = _mtl.PositionalEmbedding(input_axis=left_axis, output_axis=right_axis, name='position-embeddings') # (B, G, E) + (1, G, E)
        self._merge = _mtl.Merge(left_axis=left_axis, right_axis=right_axis, left=True, name='merge-embeddings') # (B, G, E) => (B, G * E)
        self._dense = tf.keras.layers.Dense(units=latent_dim, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', name='compress-embeddings') # (B, G * E) => (B, L), typically L = E

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return self._dense(self._merge(self._embedding(self._divide(inputs))))
```

### Detokenization Block

The `_embedding` layer is actually redundant, but I have a strong urge to make `DetokenizeBlock` the symmetric of `TokenizeBlock`...

```python
class DetokenizeBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        token_dim: int=4,
        embedding_dim: int=256,
        **kwargs
    ) -> None:
        super(DetokenizeBlock, self).__init__(**kwargs)
        # layers
        self._dense = tf.keras.layers.Dense(units=token_dim * embedding_dim, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', name='decompress-embeddings') # (B, L) => (B, G * E), typically L = E
        self._divide = _mtl.Divide(input_axis=-2, output_axis=-1, insert=True, factor=embedding_dim, name='divide-embeddings') # (B, G * E) => (B, G, E)
        self._embedding = _mtl.PositionalEmbedding(input_axis=-2, output_axis=-1, name='position-embeddings') # (B, G, E) + (1, G, E)
        self._merge = _mtl.Merge(left_axis=0, right_axis=1, left=True) # (B, G, E) => (B * G, E)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return self._merge(self._embedding(self._divide(self._dense(inputs))))
```

[github-mlqa]: https://github.com/facebookresearch/MLQA
[youtube-karpathy-tokenizer]: https://www.youtube.com/watch?v=zduSFxRajkE

[article-github-tokun-1]: https://github.com/apehex/tokun/blob/main/articles/tokun.1.md
[article-github-tokun-16]: https://github.com/apehex/tokun/blob/main/articles/tokun.16.md

[notebook-colab]: https://colab.research.google.com/github/apehex/tokun/blob/main/notebooks/tokun.4.ipynb
[notebook-github]: https://github.com/apehex/tokun/blob/main/notebooks/tokun.4.ipynb
[notebook-huggingface]: https://github.com/apehex/tokun
[notebook-kaggle]: https://github.com/apehex/tokun

[tokun-github]: https://github.com/apehex/tokun
[tokun-huggingface]: https://github.com/apehex/tokun
[tokun-kaggle]: https://github.com/apehex/tokun
