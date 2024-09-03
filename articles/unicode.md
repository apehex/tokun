# This Title Is Already Tokenized

unicode is all you need
ai encoding uses language from -1000
from hieroglyphs to 
back from the prehistoric ages
this title is already tokenized

> `tokun` took tokens to t-can

in machine learning 3 worlds / visions are at odds: the computer, math and human sides.

tokenization bridges the gap from machine to tensors using human intuition, with algorithms like BPE.

in my [previous article][huggingface-tokenization-1], I proposed to train a model to translate / compress the encoding bytes into embeddings.

Actually, none of this is necessary since any digital text has already an encoding which can directly be used as embedding.

from encoding to embedding

<img src="../.github/header.png" alt="Neural tokenization" title="Source: Image by Author and generated with MidJourney" width="100%" style="margin: auto;"/>

<img src=".images/tiktoken/russian.utf32.codes.png" width="75%" style="margin: auto;"/>

<img src=".images/tiktoken/russian.utf32.tokens.codes.png" width="75%" style="margin: auto;"/>

<img src=".images/tiktoken/russian.utf32.tokens.bytes.png" width="75%" style="margin: auto;"/>

## Intuition

Russian translation of `In simple cases, the concepts of "lexeme" and "token" are identical`:

```
В простых случаях понятия «лексема» и «токен» идентичны.
```

56 UTF-32 codepoints:

```
[1042, 32, 1087, 1088, 1086, 1089, 1090, 1099, 1093, 32, 1089, 1083, 1091, 1095, 1072, 1103, 1093, 32, 1087, 1086, 1085, 1103, 1090, 1080, 1103, 32, 171, 1083, 1077, 1082, 1089, 1077, 1084, 1072, 187, 32, 1080, 32, 171, 1090, 1086, 1082, 1077, 1085, 187, 32, 1080, 1076, 1077, 1085, 1090, 1080, 1095, 1085, 1099, 46]
```

224 UTF-32-BE bytes:

```
[0, 0, 4, 18, 0, 0, 0, 32, 0, 0, 4, 63, 0, 0, 4, 64, 0, 0, 4, 62, 0, 0, 4, 65, 0, 0, 4, 66, 0, 0, 4, 75, 0, 0, 4, 69, 0, 0, 0, 32, 0, 0, 4, 65, 0, 0, 4, 59, 0, 0, 4, 67, 0, 0, 4, 71, 0, 0, 4, 48, 0, 0, 4, 79, 0, 0, 4, 69, 0, 0, 0, 32, 0, 0, 4, 63, 0, 0, 4, 62, 0, 0, 4, 61, 0, 0, 4, 79, 0, 0, 4, 66, 0, 0, 4, 56, 0, 0, 4, 79, 0, 0, 0, 32, 0, 0, 0, 171, 0, 0, 4, 59, 0, 0, 4, 53, 0, 0, 4, 58, 0, 0, 4, 65, 0, 0, 4, 53, 0, 0, 4, 60, 0, 0, 4, 48, 0, 0, 0, 187, 0, 0, 0, 32, 0, 0, 4, 56, 0, 0, 0, 32, 0, 0, 0, 171, 0, 0, 4, 66, 0, 0, 4, 62, 0, 0, 4, 58, 0, 0, 4, 53, 0, 0, 4, 61, 0, 0, 0, 187, 0, 0, 0, 32, 0, 0, 4, 56, 0, 0, 4, 52, 0, 0, 4, 53, 0, 0, 4, 61, 0, 0, 4, 66, 0, 0, 4, 56, 0, 0, 4, 71, 0, 0, 4, 61, 0, 0, 4, 75, 0, 0, 0, 46]
```

## Notice

will start with specific issues and progressively build a new system. 

In the following sections, I have minimized the interface of [Tiktokenizer][tiktokenizer-gpt-4], but the data is still accurate.

western language
interested on perspective other culture / continent

## Tokenization And Ancient Languages

essentially, tokenization merges forms monolithic 

<img src=".images/tiktoken/russian.gpt4o.png" width="75%" style="margin: auto;"/>

20 tokens in GPT-4o:

```
[3540, 14063, 6172, 78267, 72435, 1691, 2415, 32555, 41118, 1924, 816, 2415, 338, 2533, 776, 1924, 131660, 94743, 1208, 13]
```

logographic

compose from simpler elements
=> alphabetic and syllabic

more generally this is the concept of 

## Objectives / Ideal

- reduce the sequence length: faster processing, less resources
- give "meaning"
- avoid "meaningless" predictions and constrain to

desired properties:

- compression
- proximity
- composition
- timeless: concepts and dates appear more / less frequently depending on the period

start from "blabla" => target `[[]]`

## Language Basis

- computer: sequence => codepoint => byte => bits
- math: tensors => axes => dimensions
- human: paragraph => sentence => word => symbols / letters

common denominator = the macro elements all break down into simpler parts.
while there are the number of possible macro elements grows exponantially, there are very few basis elements:

- computer: 2 bits
- human: 26 lowercase letters and a few symbols for Latin languages
- math: real numbers, actually infinite

all these schemes take advantage of the rules of combinatorics

tokenization = opposite!
base elements are 

## Representing The Predictions

Suppose GPT-4o processed the following sentence:

```
This paper was based mainly on the attention mechanism developed by Bahdanau et al. in 2014.[11]
```

```
2500, 6651, 885, 8583, 673, 316, 8400, 7557, 220, 667, 19, 57102, 17, 27794, 6340
```

output = probability of next token, by index
softmax = one true

binary error => close prediction
but, close tokens are unrelated
=> other input repr

## Pros

- standard: shared worldwide
- international: all languages are covered
- native: no training required
- compression: smallest tensor size possible
- fixed: all tokens have the same dimension, chosen freely
- structured: Unicode has 
- numbers: the encoding is correlated to actual number values
- composition: embeddings now 

## Cons

- brittle: small changes

## Input Representation

all examples: 16 characters = 16 UTF-32 codepoints = 64 UTF-32 bytes

### Features = Sequence Of Codepoints

cons:

- there are 262144 "basic" elements, similar to regular tokenizer vocabularies
- single value with delta = 1 / 0x40000 = 3.8147e-06 => little separation between codepoints

### Features = Sequence Of Bytes

### NOT binary

the 256 byte values play a specific role, while the 0 and 1 have the same meaning.

for example the byte "0" is padding.

### Features = Composite Embeddings

- byte / 256
- codepoint / 0x40000
- byte sequence = embedding index => unrelated embeddings (rather than smooth function)

## Next

LLMs are still in the stone age

why / how did tokenization last so long?

compiler + llm using tokun embeddings

better representation?

[huggingface-tokenization-1]: https://huggingface.co/blog/apehex/tokenization-is-a-dead-weight
[tiktokenizer-gpt-4]: https://tiktokenizer.vercel.app/?model=gpt-4
