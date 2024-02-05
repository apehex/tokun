# GPM: Generative Password Manager

> Stateless password manager, powered by AI tensors.

Password management is up there with cookie popups and ads, a major pain in the ass.
Here's an elegant solution using tools from the AI field.

## Features

> passwords are **never stored**, so they can't be leaked
> passwords are **never transmited**, there is no need to sync devices
> all the passwords are generated from a **single master key**

## Principle

Contrary to traditional password managers, the passwords are not saved on disk:
they are (re)generated each time.

The master-key is used to randomly fill the tensor weights, which constitute a MLP model.

This MLP model then takes the login information as input and outputs a password.

Even though the process generates high entropy passwords, it is deterministic and will always output the same password for a given login.

## CLI

Only 3 arguments are required:

- the master key
- the login target
- the login id

If they are not specified on the command line, the user will be prompted during the execution.

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

The inputs are then processed:

0. setup the hyper-parameters:
    - use the whole ASCII table as input vocabulary and save its shape
    - compose the output vocabulary and save its shape
    - cast the master-key into an integer seed
1. define the mappings between IO strings and tensors
2. preprocess / clean the string inputs
3. encode the inputs as a sequence tensor X for the MLP
4. create the model corresponding to the hyper-parameters
5. sample / generate the password as a tensor Y
6. decode the probability tensor Y into an actual password string

## 0) Setup The Hyper Parameters

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

### Defining the Input Vocabulary

### Composing The Output Vocabulary

### Casting The Master Key Into The Seed

The seed is derived from the master-key of the user.

```python
```

## Creating The MLP Model

## Preprocessing The Inputs

- site
- login

Want to:

- increase entropy to trigger different parts of the NN (context to have unique inputs)
- have  deterministic (replicable) operations
- filter out non ASCII characters
- output the same value, whether or not there's unicode characters

## Sampling = Password Generation

Issues:

- if `len(site + login) < password_length`?

## Evaluation

- randomness:
    - bad init (same seed for all tensors)
- overhead? actually a feature? (hinders bruteforcing attempts)
- security:
    - the model should not be stored in memory / disk

## Improvements

- performance:
    - use the base `numpy` instead of `tensorflow`
    - replace the model with its base weight tensors and matrix multiplications
