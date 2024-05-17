# Tokun-4x4

> can `tokun` tackle

Current tokenizers have notorious issues that are bringing all the LLMs down.

llama3 would...

A "compelling argument" as ChatGPT puts it, right?

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
- roughly dividing the attention layers by a factor ` 256 ** 2 = 65536`
- dividing the feed forward layers by the same factor
- all in all the model would be shrunk to 15M parameters

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

Still, there may be arguments for freezing the weights of `tokun`:

- sharing the model: community effort to improve the tokenizer?
- modularity:
- interpretation: common ground to compare the internals?
- normalization:
- security: having a shared referential also facilitates

## Roadmap

`tokun-4` actually solved most of the target limitations:

- [x] bla

Now we're just pushing the concept to the limit with the `4x4x4` variant `tokun-16`.

## Model

Similar to previous models, with either more or bigger `tokenize` blocks.

Push the concept / compression until the model fails to achieve 100% accuracy.

### Inputs

### Architecture

Added non causal self-attention to the blocks.

#### Hyper Parameters

#### Encoder

#### Decoder

### Outputs

## Training

### Augmentations

### Perturbations

## Results

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

Showcase the tokenizer on a full transformer model like llama3.

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
- [Hugging Face][notebook-huggingface]
- [Kaggle][notebook-kaggle]

## Implementation Details

[github-llama3]: https://github.com/meta-llama/llama3/blob/main/llama/model.py
[github-mlqa]: https://github.com/facebookresearch/MLQA
[youtube-karpathy-tokenizer]: https://www.youtube.com/watch?v=zduSFxRajkE

[article-github-tokun-1]: https://github.com/apehex/tokun/blob/main/articles/tokun.1.md
[article-github-tokun-4]: https://github.com/apehex/tokun/blob/main/articles/tokun.4.md

[notebook-colab]: https://colab.research.google.com/github/apehex/tokun/blob/main/notebooks/tokun.16.ipynb
[notebook-github]: https://github.com/apehex/tokun/blob/main/notebooks/tokun.16.ipynb
[notebook-huggingface]: https://github.com/apehex/tokun
[notebook-kaggle]: https://github.com/apehex/tokun

[tokun-github]: https://github.com/apehex/tokun
[tokun-kaggle]: https://github.com/apehex/tokun
