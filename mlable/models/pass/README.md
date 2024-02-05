# GPM: Generative Password Manager

> Stateless password manager, powered by AI tensors.

Password management is up there with cookie popups and ads, a major pain in the ass.
Here's an elegant solution using tools from the AI field.

## Features

- passwords are **never stored**, so they can't be leaked
- passwords are **never transmited**, there is no need to sync devices
- all the passwords are generated from a **single master key**
- password generation on the fly

## CLI

## Process Overview

Given:

- login
- target
- master key
- length
- vocabulary composition

0. setup: desired output vocabulary + length
1. master password => hyper parameters
2. model creation
3. input encoding
4. sampling / password generation

## Hyper Parameters

- seed
- tensor shapes
- vocabulary (alpha / numbers / symbols)
- password length

## Master Key

Master key => hyper parameters

## Model

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
