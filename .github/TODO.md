## TODO

### Common

- [x] replicate Keras modules in base Tensorflow
    - [x] initializers
    - [x] layers
    - [x] losses
    - [x] optimizers
- [x] inputs
    - [x] tokenizer operate directly on tensors instead of looping => *no*, make tokenization framework agnostic
- [x] sampling
- [x] pyTorch variants:
    - [x] MLP
    - [x] CNN
    - [x] SAT

### GPT

- [x] positional embedding:
    - [x] normalize? *no*
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
- [ ] GAN? colony of models?
- [ ] other:
    - [ ] keep track of the token's parts / parents? => enforce the similarity
    - [ ] density: encoding that actually uses the full range of uint256 / float32
- [ ] solidity:
    - [ ] specific tokens for solidity assembly
    - [ ] specific tokens for solidity source code
- [ ] view tokenization results like tiktokensizer
- [ ] properties / loss functions:
    - [ ] embedding space:
        - [ ] the token embedding is dense (contrary to one-hot encoding)
        - [ ] split & merged tokens are close to their siblings
        - [ ] synonyms are close
        - [ ] the loss for numbers varies with delta
- [ ] tokens:
    - [ ] numbers are represented as a single token:
        - [ ] integers
        - [ ] hex numbers (cute into chunks of 32 bytes = 256 bits)
        - [ ] floats in any format (scientific etc)
        - [ ] represented as a vector of length 256
    - [ ] keywords are single tokens
    - [ ] variables are special tokens ?
