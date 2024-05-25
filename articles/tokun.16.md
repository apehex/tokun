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

The encoder is a stack of `TokenizeBlock`, with added normalization and self-attention layers:

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

But, now the aim is to cover the whole Unicode space:
words, numbers and code actually cover a very limited range of the possibles.

Using standard datasets with may push the model into learning common patterns.
This may prevent the model from generalizing to new regions of the Unicode space.

The role of `tokun` is actually to compress the encoding, *not* the language.

So the most significant change in this iteration is to **train the model on random sequences** of UTF-32-BE bytes.
Since the dataset is random, there is no need for data augmentation.

## Results

For this model to be relevant, it has to be perfectly accurate so that embeddings can be reversed into their matching sequence of characters.

### Metrics

All the configurations reach 100% accuracy on the MLQA validation dataset.

However, the goal is to have `tokun` be able to encode any arbitrary 16-gram (token) of codepoints.
On the random dataset, the `4x4x4` configuration is stuck at 75% accuracy:

![][image-graph-accuracy-4x4x4]

However model configurations with larger weight tensors like `4x16` achieve 99.999% accuracy on the whole Unicode space (and 100% on MLQA).

### Embeddings

Even though the model was trained on random bytes, the embeddings show intuitive properties.

The model tends to group tokens by language:

| German                    | Vietnamese                    |
| ------------------------- | ----------------------------- |
| ![][image-16-umap-german] | ![][image-16-umap-vietnamese] |

Since German share the same alphabet as English and Spanish, it is surprising that tokens are not mixed.
Vietnamese has a more distinct wording with its accent so it is both expected and more apparent.

The encoder model performs nested tokenization, with each successive block grouping previous embeddings.
So the tokens of length 16 are made of tokens of length 4.

The embeddings for the 4-tokens can also be viewed.
Their latent space shows even more structure:

| PCA                       | UMAP                          |
| ------------------------- | ----------------------------- |
| ![][image-4-pca]          | ![][image-4-umap]             |

It is easier to make sense of the embeddings by exploring them.
A few samples were encoded and [exported to the Github repository](../embeddings/) and can be viewed with the [tensorboard projector][tensorboard-projector].

### Robustness

Contrary to the previous models, `tokun-16` is susceptible to noise:

```python
__std = tf.math.reduce_std(EMBEDDINGS[16]['en'], axis=0)
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

Even under a single standard deviation the embedding neighborhoods are unstable:

| Overview                  | Zoom                              |
| ------------------------- | --------------------------------- |
| ![][image-pca-neighbors] | ![][image-pca-neighbors-zoom]    |

This could be a deal-breaker, as it may be hard for a LLM to predict precise embeddings.

### Configurations

Performance is very dependent on the configuration.
While all models achieve 100% accuracy on MLQA, it is harder to cover the whole Unicode space:

| 4x4x4 vs 4x16                         | 16x4 vs 64                        |
| ------------------------------------- | --------------------------------- |
| ![][image-graph-accuracy-4x4x4-4x16]  | ![][image-graph-accuracy-16x4-64] |

Only the `4x16` and `16x4` are able to reach 100% on random byte sequences.
These models have more parameters:

- `4` blocks have a kernel of size `4 * 256 * 256 = 262144`
- `16` blocks have `16 * 256 * 256 = 1048576`
- `64` blocks have `64 * 256 * 256 = 4194304`

But the `64` configuration has trouble learning because it treats each combination of 64 bytes as unique while others operate on abstractions for each unit.

### Attention And Normalization

It seems that the attention layer makes no difference:

![][image-graph-accuracy-layers]

Having a normalization layer (`LayerNorm`) accelerates the training by more than two times.

### Activation

The model was trained with `tanh`, `relu` and `silu` / `swish` activation functions.

No significant impact was seen on the overall performance.
The latent space shows different patterns for each.

## Features

### Extension Of Tokun-4

`tokun-16` keeps all the features of the previous model `tokun-4`:

- it is obviously still a NN tokenizer
- it has special tokens because UTF-32-BE has special characters
- it produces vector embeddings of dimension 256
- it has 100% encode-decode accuracy on its training languages
- it is independent from the input splitting
- its tokens all have an even 16 character length
- its tokens are related to each other
- its tokens hold all the information on their parts

### Compression

The input tensor has shape `(B' * G * G * G, E)` and the embedding is `(B, E)`:
the model performs a compression by factor 64 compared to the UTF-32 bytes, or 16 wrt unicode strings.

```python
__x.shape # (B' * G * G * G, E) = (B * 4 * S, E) = (128 * 4 * 256, 256)
# (131072, 256)
__e.shape # (B', E) = (B * 4 * S / (G * G * G), E) = (128 * 4 * 256 / 64, 256)
# (2048, 256)
```

### Generalization

The variant `4x4x4` is stuck around 50% on new Unicode regions, even when trained on random data.

While the `4x16`, with its larger weight tensors, is able to operate on the whole Unicode space:

```
# INPUT ################################################################

위키백과, 우리 모두의 백과사전.
t-분포 확률적 임베딩(t-SNE)은 데이터의 차원 축소에 사용되는 기계 학습 알고리즘 중 하나로, 2002년 샘 로이스Sam Rowise와 제프리 힌튼에 의해 개발되었다.[1] t-SNE는 비선형 차원 축소 기법으로, 고차원 데이터를 특히 2, 3차원 등으로 줄여 가시화하는데에 유용하게 사용된다. 구체적으로 t-SNE는 비슷한 데이터는 근접한 2, 3차원의 지점으로, 다른 데이터는 멀리 떨어진 지점으로 맵핑한다.

# OUTPUT ###############################################################

위키백과, 우리 모두의 백과사전.
t-분포 확률적 임베딩(t-SNE)은 데이터의 차원 축소에 사용되는 기계 학습 알고리즘 중 하나로, 2002년 샘 로이스Sam Rowise와 제프리 힌튼에 의해 개발되었다.[1] t-SNE는 비선형 차원 축소 기법으로, 고차원 데이터를 특히 2, 3차원 등으로 줄여 가시화하는데에 유용하게 사용된다. 구체적으로 t-SNE는 비슷한 데이터는 근접한 2, 3차원의 지점으로, 다른 데이터는 멀리 떨어진 지점으로 맵핑한다.����

# SCORE ################################################################

1.0
```

## Next

On its own `tokun` is performing surprisingly well.
Every 4-gram of Unicode codepoints can be compressed without loss, even arbitrary byte sequences.

Now, the question is whether `tokun` integrates well within a full featured LLM.
Next we'll plug `tokun` into a custom `llama3`, `llaminate`.

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

### Random Dataset

The training dataset is made of random codepoints in the first 4 planes of Unicode:

```python
def random_codepoint(lower_plane: int=0, upper_plane: int=0x40000) -> list:
    __h = '{0:0>8x}'.format(int(random.uniform(lower_plane, upper_plane)))
    return list(bytes.fromhex(__h))

def random_sample(sample_size: int, lower_plane: int=0, upper_plane: int=0x40000) -> list:
    __nested = [random_codepoint(lower_plane=lower_plane, upper_plane=upper_plane) for _ in range(sample_size)]
    return list(itertools.chain.from_iterable(__nested))

def random_dataset(size: int, sample_size: int, lower_plane: int=0, upper_plane: int=0x40000) -> tf.data.Dataset:
    def __generator() -> iter:
        for _ in range(size):
            yield random_sample(sample_size=sample_size, lower_plane=lower_plane, upper_plane=upper_plane)
    return tf.data.Dataset.from_generator(
        generator=__generator,
        output_signature=tf.TensorSpec(shape=(4 * sample_size,), dtype=tf.int32))
```

[github-llama3]: https://github.com/meta-llama/llama3/blob/main/llama/model.py
[github-mlqa]: https://github.com/facebookresearch/MLQA
[tensorboard-projector]: https://projector.tensorflow.org/
[youtube-karpathy-tokenizer]: https://www.youtube.com/watch?v=zduSFxRajkE
[wiki-unicode-plane]: https://en.wikipedia.org/wiki/Plane_(Unicode)

[article-github-tokun-1]: https://github.com/apehex/tokun/blob/main/articles/tokun.1.md
[article-github-tokun-4]: https://github.com/apehex/tokun/blob/main/articles/tokun.4.md

[image-graph-accuracy-4x4x4]: .images/16/graph.accuracy.4x4x4.png
[image-graph-accuracy-4x4x4-4x16]: .images/16/graph.accuracy.4x16-vs-4x4x4.png
[image-graph-accuracy-16x4-64]: .images/16/graph.accuracy.16x4-vs-64.png
[image-graph-accuracy-layers]: .images/16/graph.accuracy.layers.png
[image-pca-neighbors]: .images/16/pca.neighbors.png
[image-pca-neighbors-zoom]: .images/16/pca.neighbors.zoom.png
[image-16-umap-german]: .images/16/umap.16.german.png
[image-16-umap-vietnamese]: .images/16/umap.16.vietnamese.png
[image-4-pca]: .images/16/pca.4.png
[image-4-umpa]: .images/16/umap.4.png

[notebook-colab]: https://colab.research.google.com/github/apehex/tokun/blob/main/notebooks/tokun.16.ipynb
[notebook-github]: https://github.com/apehex/tokun/blob/main/notebooks/tokun.16.ipynb
[notebook-huggingface]: https://github.com/apehex/tokun
[notebook-kaggle]: https://github.com/apehex/tokun

[tokun-github]: https://github.com/apehex/tokun
[tokun-kaggle]: https://github.com/apehex/tokun
