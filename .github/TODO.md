## TODO

### Common

- [x] replicate Keras modules in base Tensorflow
    - [x] initializers
    - [x] layers
    - [ ] losses
    - [x] optimizers
- [ ] inputs
    - [ ] tokenizer operate directly on tensors instead of looping
- [ ] sampling
- [ ] pyTorch variants:
    - [ ] MLP
    - [ ] CNN
    - [ ] SAT

### GPT

- [ ] positional embedding:
    - [ ] normalize?
- [ ] self attention layer:
    - [ ] key
    - [ ] query
    - [ ] value
    - [ ] masking + softmax
- [ ] residual block:
    - [ ] layer norm
    - [ ] self attention
    - [ ] residual