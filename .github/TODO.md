# TODO

## Objectives

- [x] dense embeddings, rather than sparse "one-hot"
- [x] guiding without fixing: no frozen dictionary, context agnostic
- [x] tokenization independant of the input partitioning / shift
- [x] dense encoding != one-hot vectors on the vocabulary
- [x] composite tokens have parent / child relation: "splitting" carries the information of "split" and "ing"
- [x] reduce token dimension: from several 100k to 256!
- [x] better support for eastern languages: Arabic, Chinese, Hindi, etc

## Dataviz

- [x] spatial repartition of tokens
- [ ] embeddings of child <=> parent tokens
- [x] limit embedding size = when fidelity starts to drop = max compression = x64?

## Curriculum

- [ ] shift training data by 1, 2, ..., G - 1 ticks along the time / context axis
- [ ] switch between equivalent formats:
    - [x] byte shift
    - [ ] abbreviations: "can't" <=> "cannot"
    - [ ] change number format (while keeping the same value)
- [ ] random perturbations on the inputs:
    - [ ] letter capitalization
    - [ ] byte replacement
    - [ ] byte insertion
    - [ ] reversing order in groups?
- [ ] equivalence 1 <=> 4 <=> 4 x 4:
    - [ ] pad data with 0s to fill bigger tokens until they match their parts

## Blocks

- [x] tokenization:
    - [x] simplify: divide + position + merge = reshape + dense (position = dense bias on the merged vector)
    - [x] move data from axis 0 to axis -1 in the end: (B * G, E) => (B, G * E)
- [x] detokenization
    - [x] simplify: same as the tokenization block
- [x] head

## Models

- [x] VAE
- [x] VAE + CNN
- [x] VAE + CNN + attention
- [x] VAE + hierarchical CNN
- [x] VAE + hierarchical CNN + attention
- [x] VAE + hierarchical CNN + attention + normalization
