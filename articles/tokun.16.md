# Tokun-4x4

> can `tokun` tackle

Current tokenizers have notorious issues that are bringing all the LLMs down.

llama3 would...

A "compelling argument" as ChatGPT puts it, right?

## Intuition

With `tokun-16`, the model is getting one step closer to the limit of input compression.

16 Unicode characters will be represented by a `float32` vector of dimension 256.
It's 1024 output bytes for every 64 input bytes.

It would appear that there is still a lot of leeway.
Actually, if there was a 1:1 match it would mean that only a single embedding value would map to each token.

It would be impractical since it would require LLMs to predict very precise values.
Suppose that `[0.025 1.52 1.67 2.24 ...]` represents `"hello world! \o/"`.
If the LLM outputs `[0.025 1.53 1.67 2.24 ...]` it may just have produced a totally random string.

In fact, $\frac{1024}{64} = 16$ is a good ratio:
we want each chunk of 16 characters to be represented by a region in the space of dimension 256.

Ultimately, we would like to map the entire Unicode space.
In theory, Unicode is made of $2^32$ code points.

However only [17 Unicode planes][wiki-unicode-plane] are used because of the limitations.
Out of those 17 planes only 7 are actually used, with 2 reserved for user customization (think empty).

So our final goal is only to map 327,680 code points.

## Applying Tokun To Current LLMs

### Model Shape

Consider [`llama3-8B`][github-llama3]:

```python
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1 # around 128k
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 500000

    max_batch_size: int = 32
    max_seq_len: int = 2048
```

Ignoring the biases and other normalizing vectors, here's a rough count of the parameters in `llama3-8B` by layer:

- embedding: `128000 * 4096 = 524288000`
- transformer block:
    - RMS normalization: `4096`
    - attention: `4 * 4096 * 4096 = 67108864`
    - feed forward: `3 * 4096 * 4 * 4096 = 201326592`
    - total: `2 * 4096 + 67108864 + 201326592 = 268443648`
- head:
    - RMS normalization: `4096`
    - linear: `4096 * 128000 = 524288000`
    - total: `4096 + 524288000`
- in total: `524288000 + 32 * 268443648 + 524292096 = 9638776832`

So 8 billion parameters in the body but 1 billion more on both ends.

Replacing the 128k vocabulary with `tokun` embeddings would at least:

- bring the embedding layer down to `256 * 4096 = 1048576`
- reduce the output layer by the same amount
- for a total of more than 1B parameters less

Also `tokun-16` is able to compress any 16 characters into vectors of dimension 256.
So it looks like an embedding dimension of 4096 is excessive, especially for a minimal model.

At most, bringing the embedding dimension to 256 would mean:

- removing the embedding layer entirely
- roughly dividing the attention layers by a factor `16 ** 2 = 256`
- dividing the feed forward layers by the same factor
- all in all the model would be shrunk to 38M parameters or so

But that's an extreme scenario, the balance is definitely somewhere in-between.

### Sequence And Context

With this `4x4x4` variant of `tokun`, the length of the input sequence is divided by 16.

It means that LLMs could fit on average 4 times more tokens per inference.

### Training

As mentioned in the [first article][article-github-tokun-1], handling meaningful embeddings (instead of arbitrary IDs) may significantly lower the data requirements.

The training with either `tokun` kept outside like regular tokenizers or incorporated into a larger model.

In the first case, since `tokun` is comparatively small, it could be trained on a vast volume of data and fast.
This guarantees that the tokenizer captures the structure of the whole Unicode space and the mechanics of many languages, code and maths included.

Then it may be merged with a LLM and fine-tuned to fit perfectly.

## Roadmap

`tokun-4` actually reached most of the target goals:

- [x] is an actual neural network
- [x] generalizes across all languages
- [x] compresses input sequences 4 times
- [x] produces embeddings of dimension 256
- [x] comes with built-in special tokens
- [x] treats all inputs evenly
- [x] keeps all the semantic information regardless of the splits
- [x] embeddings hold information on their parts up to the byte level

Now we're just pushing the concept to the limit with the `4x4x4` variant `tokun-16`.
And try to map all the Unicode planes.

## Model

The model is very similar to previous iterations, with either more or bigger `tokenize` blocks.

Most of the model is kept identical: details are available in the [first][article-github-tokun-1] and [second][article-github-tokun-4] articles.
Only the changes are described here.

### Inputs

The input is a flat array of UTF-32-BE bytes, of shape `(B' * G * G * G, 256)`.

Actually, the exact factorisation of the batch dimension depends on the model architecture:
the (de)tokenization block at depth $i$ will group $G_i$ units.

In the end, the input tensor is factored as:

$$\begin{align}
(B' * G_0 * G_1 * G_2) \text{ or more generally } (B' * \prod_{i = 0}^{D - 1} G_i)
\end{align}$$

Where $D$ is the depth of the model or the number of (de)tokenization blocks.

Typically $G_0 = G_1 = G_2 = 4$ and the model is labeled `4x4x4`.
This variant groups 64 UTF-32-BE bytes or 16 characters.

But there are other possibilities: for example, the architectures `4x16` and `2x2x2x8` have the same compression factor.
The cardinal configurations are compared in the [results section](#configurations).

It may happen happen that a `4x4x4` model is actually called `tokun-4x4`, like the title of this article.
When named `4x4x4` it is considered from the byte level, while `4x4` designates the tokenization on the character level.

I haven't decided yet on the convention.
On one hand it would be more precise to describe all the layers.
On the other hand UTF-32-BE always produces 4 bytes which form a logical unit that looks like an automatic candidate for the first layer.

### Architecture

#### Hyper Parameters

`tokun-4x4` is still a CNN VAE with the following hyper parameters:

```python
ATTENTION = True
NORMALIZATION = True

N_TOKEN_DIM = [4, 4, 4] # G, for each block
N_ENCODING_DIM = 256 # U
N_EMBEDDING_DIM = N_ENCODING_DIM # E
N_LATENT_DIM = N_EMBEDDING_DIM # L
```

The encoder and the decoder perform symmetric merge and divide operations on the batch axis of the input tensor.

#### Encoder

The encoder is a stack of `TokenizeBlock`, with added normalization and self-attention layers.

0. the `LayerNormalization` helps maintain coherence between layers
1. the `Divide` layer splits the batch axis: `(B * G, E)` => `(B, G, E)`
2. the `PositionalEmbedding` layer distinguishes each of the `G` token units with a specific bias
3. the `Attention` gives relative weights to each token unit, enabling more complex patterns
4. the `Merge` layer groups all the embeddings on the last dimension: `(B * G, E)` => `(B, G * E)`
5. the `Dense` layer finally compresses the dimension `G * E` into `E`

`token-4` had 2 such layers and `tokun-4x4` just stacks 3.

Actually, [the implementation](#implementation-details) allows any configuration:

```python
N_TOKEN_DIM = [4, 4] # tokens of length 4 characters / 16 UTF-32 bytes (tokun-4)
N_TOKEN_DIM = [4, 4, 4] # tokens of length 16 characters (tokun-4x4)
N_TOKEN_DIM = [4, 16] # alternative configuration with tokens of length 16
```

#### Decoder

The decoder is a stack of `DetokenizeBlock`

1. the `Dense` layer expands the latent embedding: `(B, E)` => `(B, G * E)`
2. the `Divide` layer splits the last axis: `(B, G * E)` => `(B, G, E)`
3. the `PositionalEmbedding` layer adds markers to each token unit
4. the `Attention` gives relative weight to each token unit
5. the `Merge` layer flattens the tensor: `(B, G, E)` => `(B * G, E)`
6. the `LayerNormalization` adds stability so that training a block doesn't dissociate it from the others

### Outputs

By definition of a VAE, the outputs have the same shape as the input.

They are filled with probabilities instead of just zeros and ones.

## Training

Once again, training / testing were performed on [MLQA][github-mlqa] to compare the performances with previous models.

But the aim is to cover the whole Unicode space:
actually words, numbers and code have cover a very limited range of the possibles.

Using standard datasets with common patterns may push the model into learning common patterns.
This may prevent the model from generalizing to new regions of the Unicode space

So the most significant change is to **train the model on random sequences** of UTF-32-BE bytes.
The role of `tokun` is actually to compress the encoding, *not* the language.

Since the dataset is random, there is no need for data augmentation.

## Results

For this model to be relevant, it has to be perfectly accurate so that embeddings can be reversed into their matching sequence of characters.

### Metrics

### Embeddings

### Robustness

```python
__std = tf.math.reduce_std(EMBEDDINGS[4]['en'], axis=0)
__noise = tf.random.normal(shape=(256,), mean=0., stddev=tf.math.reduce_mean(__std).numpy())
__x = preprocess('tokun to can tok', groups=N_TOKEN_DIM, flatten=True)
__e = MODEL._encoder(__x)
print(postprocess(MODEL._decoder(__e)))
# tokun to can tok
print(postprocess(MODEL._decoder(__e + 0.5 * __std)))
# tokun »o can tok
print(postprocess(MODEL._decoder(__e + 0.5 * __noise)))
# to³un to can t³k
```

| Overview                  | Zoom                              |
| ------------------------- | --------------------------------- |
| ![][image-tsne-neighbors] | ![][image-tsne-neighbors-zoom]    |

Contrary to the previous models, `tokun-16` is susceptible to noise.

This could be a deal-breaker, as it may be hard for a LLM to predict precise embeddings.

### Configurations

| 4x4x4 vs 4x16                         | 16x4 vs 64                        |
| ------------------------------------- | --------------------------------- |
| ![][image-graph-accuracy-4x4x4-4x16]  | ![][image-graph-accuracy-16x4-64] |

Parameters:

- 2770688 for `4x16`

## Features

### Extension

### Compression

### Generalization

```
# INPUT ################################################################

위키백과, 우리 모두의 백과사전.
t-분포 확률적 임베딩(t-SNE)은 데이터의 차원 축소에 사용되는 기계 학습 알고리즘 중 하나로, 2002년 샘 로이스Sam Rowise와 제프리 힌튼에 의해 개발되었다.[1] t-SNE는 비선형 차원 축소 기법으로, 고차원 데이터를 특히 2, 3차원 등으로 줄여 가시화하는데에 유용하게 사용된다. 구체적으로 t-SNE는 비슷한 데이터는 근접한 2, 3차원의 지점으로, 다른 데이터는 멀리 떨어진 지점으로 맵핑한다.

# OUTPUT ###############################################################

鰄烒由싼, 鮰리 梨奐畘 由싼사鸄.
t-঄壬 畕浠鸁 鲄鲠娩(t-SNE)｀ 数이阰畘 차원 岕梌狐 憬鮩夘涔 栰싄 镙습 詌고리즘 萑 じ媘筜, 2002屄 痘 ｜이겤Sam Rowise筀 鸜锄리 傌妼꧐ 쁘해 娜娜夘峈￤.[1] t-SNE涔 ♄栠甕 차闐 岕梌 細鲕＼｜, 고차원 数이阰浼 妹傈 2, 3차원 篱甼｜ 萄壬 踀시罔镘羔豰峐 유ᢩ镘ƌ 사ᢩ夜篤. 구骴ā＼｜ t-SNE妔 畄获ぜ 数이鐰妔 狼耑镜 2, 3섨원鱘 지耐甼筜, 淤･ 数이鐰涔 奀리 媨導懄 지渐＼｜ 淵镑彜￤.

# SCORE ################################################################

0.5158730158730159
```

## Next

make a full transformer on top of `tokun`: llaminate?

- sharing the model: community effort to improve the tokenizer?
- modularity:
- interpretation: common ground to compare the internals?
- normalization:
- security: having a shared referential also facilitates

## Resources

Other articles in the serie:

- [`tokun-1`][article-github-tokun-1]
- [`tokun-4`][article-github-tokun-4]

All the variants of the model are already available on:

- [Github][tokun-github]
- [Kaggle][tokun-kaggle]

You will also find notebooks on:

- [Github][notebook-github]
- [Google Colab][notebook-colab]

## Implementation Details

All the layers now define `get_config` and `from_config` to enable the exporting features in Keras.

Apart from that, the `PositionalEmbedding`, `Divide` and `Merge` layers have the same logic as the previous iteration.

### Tokenization Block

```python
@tf.keras.saving.register_keras_serializable(package='blocks')
class TokenizeBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        left_axis: int=-2,
        right_axis: int=-1,
        token_dim: int=4,
        latent_dim: int=256,
        attention: bool=False,
        normalization: bool=False,
        **kwargs
    ) -> None:
        super(TokenizeBlock, self).__init__(**kwargs)
        # layers
        self._normalization = tf.keras.layers.LayerNormalization(axis=-1, epsilon=0.001, center=True, scale=True, name='normalization') if normalization else None # normalize each token unit independently
        self._divide = Divide(input_axis=0, output_axis=1, factor=token_dim, insert=True, name='group') # (B * G, E) => (B, G, E)
        self._embedding = PositionalEmbedding(input_axis=left_axis, output_axis=right_axis, name='position') # (B, G, E) + (1, G, E)
        self._attention = tf.keras.layers.Attention(use_scale=False, score_mode='dot', dropout=0., seed=None, name='attention') if attention else None # (B, G, E) + (B, G, E) * (B, E, G) * (B, G, E)
        self._merge = Merge(left_axis=left_axis, right_axis=right_axis, left=True, name='merging') # (B, G, E) => (B, G * E)
        self._dense = tf.keras.layers.Dense(units=latent_dim, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', name='compression') # (B, G * E) => (B, L), typically L = E

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        __t = self._normalization(inputs) if self._normalization else inputs
        __t = self._embedding(self._divide(__t))
        __t = self._attention([__t, __t, __t], return_attention_scores=False, use_causal_mask=False) if self._attention else __t
        return self._dense(self._merge(__t))

    def get_config(self) -> dict:
        __parent_config = super(TokenizeBlock, self).get_config()
        __child_config = {
            'left_axis': self._merge._left_axis,
            'right_axis': self._merge._right_axis,
            'token_dim': self._divide._factor,
            'latent_dim': self._dense.units,
            'attention': bool(self._attention),
            'normalization': bool(self._normalization)}
        return {**__parent_config, **__child_config}

    @classmethod
    def from_config(cls, config) -> tf.keras.layers.Layer:
        return cls(**config)
```

### Detokenization Block

```python
@tf.keras.saving.register_keras_serializable(package='blocks')
class DetokenizeBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        token_dim: int=4,
        embedding_dim: int=256,
        attention: bool=False,
        normalization: bool=False,
        **kwargs
    ) -> None:
        super(DetokenizeBlock, self).__init__(**kwargs)
        # layers
        self._dense = tf.keras.layers.Dense(units=token_dim * embedding_dim, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', name='decompression') # (B, L) => (B, G * E), typically L = E
        self._divide = Divide(input_axis=-2, output_axis=-1, insert=True, factor=embedding_dim, name='split') # (B, G * E) => (B, G, E)
        self._embedding = PositionalEmbedding(input_axis=-2, output_axis=-1, name='position') # (B, G, E) + (1, G, E)
        self._attention = tf.keras.layers.Attention(use_scale=False, score_mode='dot', dropout=0., seed=None, name='attention') if attention else None # (B, G, E) + (B, G, E) * (B, E, G) * (B, G, E)
        self._merge = Merge(left_axis=0, right_axis=1, left=True) # (B, G, E) => (B * G, E)
        self._normalization = tf.keras.layers.LayerNormalization(axis=-1, epsilon=0.001, center=True, scale=True, name='normalization') if normalization else None # normalize each token unit independently

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        __t = self._embedding(self._divide(self._dense(inputs)))
        __t = self._attention([__t, __t, __t], return_attention_scores=False, use_causal_mask=False) if self._attention else __t
        __t = self._merge(__t)
        return self._normalization(__t) if self._normalization else __t

    def get_config(self) -> dict:
        __parent_config = super(DetokenizeBlock, self).get_config()
        __child_config = {
            'token_dim': self._dense.units // self._divide._factor,
            'embedding_dim': self._divide._factor,
            'attention': bool(self._attention),
            'normalization': bool(self._normalization)}
        return {**__parent_config, **__child_config}

    @classmethod
    def from_config(cls, config) -> tf.keras.layers.Layer:
        return cls(**config)
```

### Encoder

```python
@tf.keras.saving.register_keras_serializable(package='models')
class Encoder(tf.keras.models.Model):
    def __init__(self, token_dim: list, encoding_dim: int, embedding_dim: int, latent_dim: int, batch_dim: int=None, attention: bool=False, normalization: bool=False, **kwargs) -> None:
        super(Encoder, self).__init__(**kwargs)
        self._encoder = tf.keras.Sequential([
            tf.keras.Input(shape=(encoding_dim,), batch_size=batch_dim, name='input'), # (B * G ^ D, U)
            tf.keras.layers.Dense(units=embedding_dim, activation=None, use_bias=False, kernel_initializer='glorot_uniform', bias_initializer=None, name='embed-1'),] # (B * G ^ D, U) => (B * G ^ D, E)
            + [TokenizeBlock(left_axis=-2, right_axis=-1, token_dim=__g, latent_dim=latent_dim, attention=attention, normalization=normalization, name='tokenize-{}_{}'.format(__g, __i)) for __i, __g in enumerate(token_dim)]) # (B * G ^ i, E) => (B * G ^ (i-1), E)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        return self._encoder(x)

    def get_config(self) -> dict:
        __parent_config = super(Encoder, self).get_config()
        __input_shape = list(self._encoder.inputs[0].shape)
        __embedding_config = self._encoder.layers[0].get_config()
        __tokenizer_config = self._encoder.layers[1].get_config()
        __token_dim = [__b.get_config().get('token_dim', 4) for __b in self._encoder.layers[1:]]
        __child_config = {
            'batch_dim': __input_shape[0],
            'encoding_dim': __input_shape[-1],
            'embedding_dim': __embedding_config['units'],
            'token_dim': __token_dim,
            'latent_dim': __tokenizer_config['latent_dim'],
            'attention': __tokenizer_config['attention'],
            'normalization': __tokenizer_config['normalization'],}
        return {**__parent_config, **__child_config}

    @classmethod
    def from_config(cls, config) -> tf.keras.layers.Layer:
        return cls(**config)
```

### Decoder

```python
@tf.keras.saving.register_keras_serializable(package='models')
class Decoder(tf.keras.models.Model):
    def __init__(self, token_dim: list, encoding_dim: int, embedding_dim: int, latent_dim: int, batch_dim: int=None, attention: bool=False, normalization: bool=False, **kwargs) -> None:
        super(Decoder, self).__init__(**kwargs)
        self._decoder = tf.keras.Sequential(
            [tf.keras.Input(shape=(latent_dim,), batch_size=batch_dim, name='input')] # (B, E)
            + [DetokenizeBlock(token_dim=__g, embedding_dim=embedding_dim, attention=attention, normalization=normalization, name='detokenize-{}_{}'.format(__g, __i)) for __i, __g in enumerate(token_dim)] # (B * G ^ i, E) => (B * G ^ (i+1), E)
            + [HeadBlock(encoding_dim=encoding_dim, name='project-head')]) # (B * G ^ D, E) => (B * G ^ D, U)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        return self._decoder(x)

    def get_config(self) -> dict:
        __parent_config = super(Decoder, self).get_config()
        __input_shape = list(self._decoder.inputs[0].shape)
        __detokenizer_config = self._decoder.layers[0].get_config()
        __head_config = self._decoder.layers[-1].get_config()
        __token_dim = [__b.get_config().get('token_dim', 4) for __b in self._encoder.layers[:-1]]
        __child_config = {
            'batch_dim': __input_shape[0],
            'latent_dim': __input_shape[-1],
            'encoding_dim': __head_config['encoding_dim'],
            'token_dim': __detokenizer_config['token_dim'],
            'embedding_dim': __detokenizer_config['embedding_dim'],
            'attention': __detokenizer_config['attention'],
            'normalization': __detokenizer_config['normalization'],}
        return {**__parent_config, **__child_config}

    @classmethod
    def from_config(cls, config) -> tf.keras.layers.Layer:
        return cls(**config)
```

### VAE

```python
@tf.keras.saving.register_keras_serializable(package='models')
class AutoEncoder(tf.keras.models.Model):
    def __init__(self, token_dim: list, encoding_dim: int, embedding_dim: int, latent_dim: int, batch_dim: int=None, attention: bool=False, normalization: bool=False, **kwargs) -> None:
        super(AutoEncoder, self).__init__(**kwargs)
        self._encoder = Encoder(token_dim=token_dim, encoding_dim=encoding_dim, embedding_dim=embedding_dim, latent_dim=latent_dim, batch_dim=batch_dim, attention=attention, normalization=normalization)
        self._decoder = Decoder(token_dim=token_dim, encoding_dim=encoding_dim, embedding_dim=embedding_dim, latent_dim=latent_dim, batch_dim=batch_dim, attention=attention, normalization=normalization)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        return self._decoder(self._encoder(x))

    def get_config(self) -> dict:
        __parent_config = super(AutoEncoder, self).get_config()
        __encoder_config = self._encoder.get_config()
        return {**__encoder_config, **__parent_config}

    @classmethod
    def from_config(cls, config) -> tf.keras.layers.Layer:
        return cls(**config)
```

[github-llama3]: https://github.com/meta-llama/llama3/blob/main/llama/model.py
[github-mlqa]: https://github.com/facebookresearch/MLQA
[youtube-karpathy-tokenizer]: https://www.youtube.com/watch?v=zduSFxRajkE
[wiki-unicode-plane]: https://en.wikipedia.org/wiki/Plane_(Unicode)

[article-github-tokun-1]: https://github.com/apehex/tokun/blob/main/articles/tokun.1.md
[article-github-tokun-4]: https://github.com/apehex/tokun/blob/main/articles/tokun.4.md

[image-tsne-neighbors]: .images/16/pca.neighbors.png
[image-tsne-neighbors-zoom]: .images/16/pca.neighbors.zoom.png

[notebook-colab]: https://colab.research.google.com/github/apehex/tokun/blob/main/notebooks/tokun.16.ipynb
[notebook-github]: https://github.com/apehex/tokun/blob/main/notebooks/tokun.16.ipynb
[notebook-huggingface]: https://github.com/apehex/tokun
[notebook-kaggle]: https://github.com/apehex/tokun

[tokun-github]: https://github.com/apehex/tokun
[tokun-kaggle]: https://github.com/apehex/tokun
