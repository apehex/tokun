# GPM: Generative Password Manager

> Stateless password manager, powered by ML neural networks.

Password management is up there with cookie popups and ads, a major pain in the ass.

Turns out you don't need to *manage* passwords, they can all be derived from a single master key.
Here's an elegant implementation using tools from the AI field.

## Features

> Passwords are **never stored**, so they can't be leaked

> Passwords are **never transmitted**, there is no need to sync devices

> All the passwords are generated from a **single master key**

## Principle

Contrary to traditional password managers, the passwords are not saved on disk:
they are (re)generated each time.

The master key is encoded and used as seed to initiate the random number generator.
Thanks to this generator, the tensor weights of a MLP are filled.

This MLP model then takes the login information as input and outputs a password.

Even though the process generates high entropy passwords, it is deterministic and will always output the same password for a given login.

## Usage

Only 3 arguments are required:

```shell
python  mlable/models/gpm/main.py --key 'never seen before combination of letters' --target 'http://example.com' --id 'user@e.mail'
# YRLabEDKqWQrN6JF
```

- the master key
- the login target
- the login id

If they are not specified on the command line, the user will be prompted during the execution:

```shell
python  mlable/models/gpm/main.py
# > Master key:
# never seen before combination of letters
# > Login target:
# http://example.com
# > Login id:
# user@mail.com
```

The full list of parameters is the following:

```shell
Generate / retrieve the password matching the input information

optional arguments:
  -h, --help                                    show this help message and exit
  --key MASTER_KEY, -k MASTER_KEY               the master key (all ASCII)
  --target LOGIN_TARGET, -t LOGIN_TARGET        the login target (URL, IP, name, etc)
  --id LOGIN_ID, -i LOGIN_ID                    the login id (username, email, etc)
  --length PASSWORD_LENGTH, -l PASSWORD_LENGTH  the length of the password (default 16)
  --nonce PASSWORD_NONCE, -n PASSWORD_NONCE     the nonce of the password
  --lower, -a                                   exclude lowercase letters from the password
  --upper, -A                                   exclude uppercase letters from the password
  --digits, -d                                  exclude digits from the password
  --symbols, -s                                 include symbols in the password
```

## Process Overview

The user provides:

- a master key
- the login informations:
    - target URL
    - user ID
- the password properties:
    - its length
    - the composition of its vocabulary (upper / lower letters, numbers, symbols)
    - a nonce, to allow multiple passwords per website

These inputs are then processed:

0. setup the hyper-parameters:
    - use the whole ASCII table as input vocabulary and save its shape
    - compose the output vocabulary and save its shape
    - cast the master key into an integer seed
1. preprocess / clean the string inputs:
    - remove unwanted characters
    - normalize the strings
2. encode the inputs as a sequence tensor X for the MLP:
    - map the input characters to integers
    - add entropy to avoid repetitions in the output
    - format as a tensor
3. create the model corresponding to the hyper-parameters
4. sample / generate the password as a tensor Y
5. decode the probability tensor Y into an actual password string

## 0. Setup The Hyper Parameters

The generative function is a MLP: it is defined by hyper-parameters.

- the seed for the random number generators
- the tensor shapes
- the input vocabulary (all the ASCII characters)
- the output vocabulary (alpha / numbers / symbols)
- the password length, which is the length of the sampling

Some of these are fixed:

```python
# size of the input / output vocabularies
N_INPUT_DIM = len(INPUT_VOCABULARY)
N_OUTPUT_DIM = N_INPUT_DIM
# shapes of the inner layers of the MLP 
N_CONTEXT_DIM = 8
N_EMBEDDING_DIM = 128
# default properties of the password
N_PASSWORD_DIM = 16
N_PASSWORD_NONCE = 1
```

Only `N_OUTPUT_DIM`, `N_PASSWORD_DIM` and `N_PASSWORD_NONCE` can be overwritten by the user.

### 0.1. Defining the Input Vocabulary

The inputs are projected on the ASCII table, all unicode characters are ignored.

This vocabulary is fixed, whatever the user typed:

```python
INPUT_VOCABULARY = ''.join(chr(__i) for __i in range(128))
```

### 0.2. Composing The Output Vocabulary

The output vocabulary dictates the composition of the model output, IE the password.

This vocabulary can contain:

- lowercase letters
- uppercase letters
- digits
- ASCII symbols, apart from the quotes `"` and `'`

```python
VOCABULARY_ALPHA_UPPER = ''.join(chr(__i) for __i in range(65, 91))                             # A-Z
VOCABULARY_ALPHA_LOWER = VOCABULARY_ALPHA_UPPER.lower()                                         # a-z
VOCABULARY_NUMBERS = '0123456789'                                                               # 0-9
VOCABULARY_SYMBOLS = ''.join(chr(__i) for __i in range(33, 48) if chr(__i) not in ["'", '"'])   # !#$%&\()*+,-./
```

It is generated from the user preferences with:

```python
def compose(lower: bool=True, upper: bool=True, digits: bool=True, symbols: bool=False) -> str:
    return sorted(set(lower * VOCABULARY_ALPHA_LOWER + upper * VOCABULARY_ALPHA_UPPER + digits * VOCABULARY_NUMBERS + symbols * VOCABULARY_SYMBOLS))
```

By default it is:

```python
''.join(compose(1, 1, 1, 0))
# '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
```

Another possibility would be to form the password out of whole words.

### 0.3. Casting The Master Key Into The Seed

A naive approach is to interpret the master key as a HEX sequence, then cast into the integer seed:

```python
def seed(key: str) -> int:
    __key = ''.join(__c for __c in key if ord(__c) < 128) # keep only ASCII characters
    return int(bytes(__key, 'utf-8').hex(), 16) % (2 ** 32) # dword
```

This doesn't work though:

```python
seed('never seen before combination of letters')
# 1952805491
seed('combination of letters')
# 1952805491
b'combination of letters'.hex()
# '636f6d62696e6174696f6e206f66206c657474657273'
```

The encoding of the string `'combination of letters'` requires 22 bytes, so it is greater than `2 ** 168`.
Prepending a prefix means adding a number times `2 ** 176` which leads to the same value modulo `2 ** 32`.

To separate the encoding of similar mater keys, it is first hashed using `sha256`:

```python
def seed(key: str) -> int:
    __key = ''.join(__c for __c in key if ord(__c) < 128) # keep only ASCII characters
    __hash = hashlib.sha256(string=__key.encode('utf-8')).hexdigest()
    return int(__hash[:8], 16) # take the first 4 bytes: the seed is lower than 2 ** 32
```

Now:

```python
seed('never seen before combination of letters')
# 3588870616
seed('combination of letters')
# 3269272188
```

## 1. Preprocessing The Inputs

The inputs are the login information for which the user wants a password:

- login target
- login id

Before being handled to the model, they need to be preprocessed to guarantee that the output matches the user expectations.

### 1.1. Removing Unwanted Characters

First, the inputs should be cleaned to:

- remove spaces: they serve no purpose and are typos like `http://example. com` 
- remove unicode characters: many typos produce invisible control characters like `chr(2002)`

Spaces can be removed with:

```python
def remove_spaces(text: str) -> str:
    return text.replace(' ', '').replace('\t', '')
```

While the encoding function detailed below will automatically replace characters outside of the input vocabulary (ASCII table) with the default character of index 0.

### 1.2. Normalizing The Strings

Several variants can be used to point to the same service:

```
example.com
https://example.com
https://example.com/
ExamPLE.COM
```

So they need to be normalized with:

```python
def remove_prefix(text: str) -> str:
    __r = r'^((?:ftp|https?):\/\/)'
    return re.sub(pattern=__r, repl='', string=text, flags=re.IGNORECASE)

def remove_suffix(text: str) -> str:
    __r = r'(\/+)$'
    return re.sub(pattern=__r, repl='', string=text, flags=re.IGNORECASE)
```

In the end:

```python
def preprocess(target: str, login: str) -> list:
    __left = remove_suffix(text=remove_prefix(text=remove_spaces(text=target.lower())))
    __right = remove_spaces(text=login.lower())
    return __left + '|' + __right
```

```python
preprocess(target='example.com', login='user')
# 'example.com|user'
preprocess(target='https://example.com', login='user')
# 'example.com|user'
preprocess(target='example.com/', login='USER')
# 'example.com|user'
```

## 2. Encoding The Inputs

### 2.1. Mapping The Characters To Integers

The mapping between character and integer is a straightforward enumeration:

```python
def mappings(vocabulary: list) -> dict:
    __itos = {__i: __c for __i, __c in enumerate(vocabulary)}
    __stoi = {__c: __i for __i, __c in enumerate(vocabulary)}
    # blank placeholder
    __blank_c = __itos[0] # chr(0)
    __blank_i = 0
    # s => i
    def __encode(c: str) -> int:
        return __stoi.get(c, __blank_i)
    # i => s
    def __decode(i: int) -> str:
        return __itos.get(i, __blank_c)
    # return both
    return {'encode': __encode, 'decode': __decode}
```

It will remove all the characters outside the input vocabulary, EG unicode characters.

### 2.2. Adding Entropy

With a character level embedding the input tensor would look like:

```python
array([101, 120,  97, 109, 112, 108, 101,  46,  99, 111, 109, 124, 117, 115, 101, 114], dtype=int32)
```

Which means that *each repetition in the input would also yield a repetition in the output password*.

Just like regular transformer models, using a context as input will make each sample more unique.
Instead of a single character, a sample is now composed of the N latest characters:

```python
array([[  0,   0,   0,   0,   0,   0,   0,   0],
       [  0,   0,   0,   0,   0,   0,   0, 101],
       [  0,   0,   0,   0,   0,   0, 101, 120],
       [  0,   0,   0,   0,   0, 101, 120,  97],
       [  0,   0,   0,   0, 101, 120,  97, 109],
       [  0,   0,   0, 101, 120,  97, 109, 112],
       [  0,   0, 101, 120,  97, 109, 112, 108],
       [  0, 101, 120,  97, 109, 112, 108, 101],
       [101, 120,  97, 109, 112, 108, 101,  46],
       [120,  97, 109, 112, 108, 101,  46,  99],
       [ 97, 109, 112, 108, 101,  46,  99, 111],
       [109, 112, 108, 101,  46,  99, 111, 109],
       [112, 108, 101,  46,  99, 111, 109, 124],
       [108, 101,  46,  99, 111, 109, 124, 117],
       [101,  46,  99, 111, 109, 124, 117, 115],
       [ 46,  99, 111, 109, 124, 117, 115, 101]], dtype=int32)
```

This can still be improved.
As long as the process is deterministic, the input can be modified in any way.

For example, the successive ordinal values can be accumulated:

```python
def accumulate(x: int, y: int, n: int) -> int:
    return (x + y) % n
```

The modulo guarantees that the encoding stays within the range of the ASCII encoding:

```python
__func = lambda __x, __y: accumulate(x=__x, y=__y + N_PASSWORD_NONCE, n=N_INPUT_DIM)
list(itertools.accumulate(iterable=__source, func=__func))
# [101, 94, 64, 46, 31, 12, 114, 33, 5, 117, 99, 96, 86, 74, 48, 35]
```

Also the context can start from the current index, instead of ending on it.
Finally the encoded input can be cycled through to create and infinite iterator:

```python
def feed(source: list, nonce: int, dimension: int) -> iter:
    __func = lambda __x, __y: accumulate(x=__x, y=__y + nonce, n=dimension) # add entropy by accumulating the encodings
    return itertools.accumulate(iterable=itertools.cycle(source), func=__func) # infinite iterable
```

This will allow to create passwords longer than the input text.

### 2.3. Formatting As A Tensor

Finally, the iterator of encoded inputs is used to generate the tensor X:

```python
def tensor(feed: 'Iterable[int]', length: int, context: int) -> tf.Tensor:
    __x = [[next(feed) for _ in range(context)] for _ in range(length)]
    return tf.constant(tf.convert_to_tensor(value=__x, dtype=tf.dtypes.int32))
```

This tensor has shape `(N_PASSWORD_LENGTH, N_CONTEXT_DIM)`:

```python
tensor(feed=__feed, length=N_PASSWORD_DIM, context=N_CONTEXT_DIM)
# <tf.Tensor: shape=(16, 8), dtype=int32, numpy=
# array([[101,  94,  64,  46,  31,  12, 114,  33],
#        [  5, 117,  99,  96,  86,  74,  48,  35],
#        [  9,   2, 100,  82,  67,  48,  22,  69],
#        [ 41,  25,   7,   4, 122, 110,  84,  71],
#        [ 45,  38,   8, 118, 103,  84,  58, 105],
#        [ 77,  61,  43,  40,  30,  18, 120, 107],
#        [ 81,  74,  44,  26,  11, 120,  94,  13],
#        [113,  97,  79,  76,  66,  54,  28,  15],
#        [117, 110,  80,  62,  47,  28,   2,  49],
#        [ 21,   5, 115, 112, 102,  90,  64,  51],
#        [ 25,  18, 116,  98,  83,  64,  38,  85],
#        [ 57,  41,  23,  20,  10, 126, 100,  87],
#        [ 61,  54,  24,   6, 119, 100,  74, 121],
#        [ 93,  77,  59,  56,  46,  34,   8, 123],
#        [ 97,  90,  60,  42,  27,   8, 110,  29],
#        [  1, 113,  95,  92,  82,  70,  44,  31]], dtype=int32)>
```

Even though the input strings `'example.com|user'` had repetitions ("e" and "m") no two lines of the tensor are the same.

The process detailed here will always produce the same tensor X.

## 3. Creating The MLP Model

Now that all the hyper-parameters are set, creating the MLP is just a formality:

```python
def create_model(
    seed: int,
    n_input_dim: int,
    n_output_dim: int,
    n_context_dim: int=N_CONTEXT_DIM,
    n_embedding_dim: int=N_EMBEDDING_DIM,
) -> tf.keras.Model:
    __model = tf.keras.Sequential()
    # initialize the weights
    __embedding_init = tf.keras.initializers.GlorotNormal(seed=seed)
    __dense_init = tf.keras.initializers.GlorotNormal(seed=(seed ** 2) % (2 ** 32)) # different values
    # embedding
    __model.add(tf.keras.layers.Embedding(input_dim=n_input_dim, output_dim=n_embedding_dim, embeddings_initializer=__embedding_init, name='embedding'))
    # head
    __model.add(tf.keras.layers.Reshape(target_shape=(n_context_dim * n_embedding_dim,), input_shape=(n_context_dim, n_embedding_dim), name='reshape'))
    __model.add(tf.keras.layers.Dense(units=n_output_dim, activation='tanh', use_bias=False, kernel_initializer=__dense_init, name='head'))
    __model.add(tf.keras.layers.Softmax(axis=-1, name='softmax'))
    # compile
    __model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False, label_smoothing=0., axis=-1, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE, name='loss'))
    return __model
```

For the purpose of this POC we are using Tensorflow and Keras, but it could actually be done with basic matrix multiplications.
Numpy would be almost as convenient to use and yield the same result.

## 4. Sampling = Password Generation

The forward pass of the tensor X in the above model will result in the probabilities for each character in the output vocabulary.

This can be directly decoded as a string like this:

```python
def password(model: tf.keras.Model, x: tf.Tensor, itos: callable) -> str:
    __y = tf.squeeze(model(x, training=False))
    __p = list(tf.argmax(__y, axis=-1).numpy())
    return _miv.decode(__p, itos=itos)
```

## Evaluation

All the operations are pieced together in the `process` function.

We can fix the internal parameters of the model like so:

```python
_process = functools.partial(
    process,
    password_length=32,
    password_nonce=1,
    include_lower=True,
    include_upper=True,
    include_digits=True,
    include_symbols=False,
    input_vocabulary=INPUT_VOCABULARY,
    model_context_dim=N_CONTEXT_DIM,
    model_embedding_dim=N_EMBEDDING_DIM)
```

Which makes it easier to test the password generation:

```python
_process(master_key='test', login_target='example.com', login_id='user')
# 'AfBOO0MGvFGikU2ZBVleuXDUFQpgR4Zg'
_process(master_key='test', login_target='http://example.com', login_id='USER')
# 'AfBOO0MGvFGikU2ZBVleuXDUFQpgR4Zg'
```

As expected the whole process is deterministic:
calls with equivalent inputs will always yield the same password, there is no need to save it.

```python
_process(master_key='verysecretpassphrase', login_target='example.com', login_id='u s e r@EMAIL.COM')
# '4ZUHYALvuXvcSoS1p9j7R64freclXKvf'
_process(master_key='verysecretpassphrase', login_target='HTTPS://example.com/', login_id='user@email.com')
# '4ZUHYALvuXvcSoS1p9j7R64freclXKvf'
```

## Improvements

This POC could be turned into a full-fledged product with:

- performance improvements:
    - use the base `numpy` instead of `tensorflow`
    - replace the model with its base weight tensors and matrix multiplications
- more output options:
    - generate the password as a bag of words
    - create whole sentences / quotes
    - force the use of certain characters / sub-vocabularies (like the symbols)
- an actual distribution as:
    - browser extension
    - binary executable (CLI)
    - mobile app
