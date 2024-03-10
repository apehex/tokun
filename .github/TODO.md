## TODO

### Common

- [x] replicate Keras modules in base Tensorflow
    - [x] initializers
    - [x] layers
    - [x] losses
    - [x] optimizers
- [x] inputs
    - [ ] tokenizer operate directly on tensors instead of looping
- [x] sampling
- [ ] pyTorch variants:
    - [x] MLP
    - [x] CNN
    - [ ] SAT

### GPT

- [ ] positional embedding:
    - [ ] normalize?
- [x] self attention layer:
    - [x] key
    - [x] query
    - [x] value
    - [x] masking + softmax
- [x] residual block:
    - [x] layer norm
    - [x] self attention
    - [x] residual

### Tokenizers

- [x] ngrams
- [x] BPE
    - [x] basic merging
    - [ ] split the data into chunks of similar (uni)codes that can be merged
- [ ] autoencoder:
    - [ ] from UTF-8?
    - [ ] avoid the overlap between successive tokens (shared information)
    - [ ] dimensionality reduction
    - [ ] losses:
        - [ ] synonyms are close
        - [ ] variations are close (plural, upper / lower case, tense)
- [ ] other:
    - [ ] keep track of the token's parts / parents? => enforce the similarity
    - [ ] density: encoding that actually uses the full range of uint256 / float32
