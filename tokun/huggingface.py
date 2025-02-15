import copy
import json

import transformers

# UTF-32-BE TOKENIZATION ######################################################

class ByteTokenizer(transformers.PreTrainedTokenizer):
    """
    Special tokenizer that encodes text as a sequence of byte indexes.

    It behaves like a regular tokenizer, with a vocabulary of 256 (1 byte, 8 bits).
    Most of the logic is inherited from `PreTrainedTokenizer`.

    The special tokens are set to obsolete / unused Unicode codepoints.
    These codepoints have legacy meanings that can be leveraged.

    UTF-8 encoding is the most compact encoding scheme, but the number of
    bytes to encode a given character can very between 1 and 4.

    UTF-32-BE encoding is very sparse since it systematically represents
    characters with 4 bytes, most of which are 0.
    However, the fixed size allows to patch the input without overlap.

    The vocabulary is set for compatibility with the parent classes.
    In UTF-32-BE "a" is '\u0061' or `[0, 0, 0, 97]`.
    The vocabulary is used to convert the indexes back and forth but the
    actual character represented comes from the combination of indexes.

    Args:
        encoding (`str, defaults to 'utf-8'):
            The text to integer mapping used.
            'utf-8' is the popular choice, but for ML purposes 'utf-32-be' is recommanded.
        bos_token (`str` or `tokenizers.AddedToken`, *optional*, defaults to '\u0002'):
            A special token representing the beginning of a sentence.
            Defaults to '\u0002', which the unicode codepoint for "start of text".
        eos_token (`str` or `tokenizers.AddedToken`, *optional*, defaults to '\u0003'):
            A special token representing the end of a sentence.
            Defaults to '\u0003', which the unicode codepoint for "end of text".
        unk_token (`str` or `tokenizers.AddedToken`, *optional*, defaults to '\u0000'):
            A special token representing an out-of-vocabulary token.
            Defaults to '\u0000', which the unicode codepoint for "null".
        sep_token (`str` or `tokenizers.AddedToken`, *optional*, defaults to '\u001e'):
            A special token separating two different sentences in the same input.
            Defaults to '\u001e', which the unicode codepoint for "record separator".
        pad_token (`str` or `tokenizers.AddedToken`, *optional*, defaults to '\u0080'):
            A special token used to make arrays of tokens the same size for batching purpose.
            Will then be ignored by attention mechanisms or loss computation.
            Defaults to '\u0080', which the unicode codepoint for "padding character".
        cls_token (`str` or `tokenizers.AddedToken`, *optional*, defaults to '\u001d'):
            A special token representing the class of the input (used by BERT for instance).
            Defaults to '\u001d', which the unicode codepoint for "group separator".
        mask_token (`str` or `tokenizers.AddedToken`, *optional*, defaults to '\u001a'):
            A special token representing a masked token.
            Defaults to '\u001a', which the unicode codepoint for "substitute".
    """

    def __init__(
        self,
        encoding: str='utf-8', # popular, but utf-32-be is recommanded
        bos_token: str='\u0002', # unicode "start of text"
        eos_token: str='\u0003', # unicode "end of text"
        unk_token: str='\u0000', # unicode "null"
        sep_token: str='\u001e', # unicode "record separator"
        pad_token: str='\u0080', # unicode "padding character"
        cls_token: str='\u001d', # unicode "group separator"
        mask_token: str='\u001a', # unicode "substitute"
        **kwargs,
    ) -> None:
        __kwargs = copy.deepcopy(kwargs)
        # properties
        self._encoding = encoding
        # enforce defaults
        __kwargs['additional_special_tokens'] = None # use the built-in special characters from Unicode
        __kwargs['split_special_tokens'] = True # in UTF-32, split the special codepoints into 4 bytes too
        # init
        super(ByteTokenizer, self).__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            **__kwargs)

    def _tokenize(self, text: str, **kwargs) -> list:
        return list(chr(__b) for __b in text.encode(self._encoding))

    def _convert_token_to_id(self, token: str) -> int:
        return ord(token)

    def _convert_id_to_token(self, index: int) -> str:
        return chr(index)

    def convert_tokens_to_string(self, tokens: iter) -> str:
        return bytes(ord(__c) for __c in tokens).decode(self._encoding, errors="ignore")

    @property
    def vocab_size(self) -> int:
        return 256

    def get_vocab(self) -> dict: # for compatibility
        return {chr(__i): __i for __i in range(256)}

    def save_vocabulary(self, save_directory: str, **kwargs) -> tuple: # for compatibility
        __prefix = kwargs.get('filename_prefix', '')
        __path = "{}/{}vocab.json".format(save_directory, __prefix or '')
        with open(__path, "w") as __file:
            json.dump(self.get_vocab(), __file)
        return (__path,)

    def build_inputs_with_special_tokens(self, token_ids_0: list, token_ids_1: list=None) -> list:
        __bos_ids = [ord(__t) for __t in self._tokenize(self.bos_token)]
        __eos_ids = [ord(__t) for __t in self._tokenize(self.eos_token)]
        __cls_ids = [ord(__t) for __t in self._tokenize(self.cls_token)]
        __token_ids = token_ids_0 + __cls_ids + token_ids_1 if token_ids_1 else token_ids_0
        return __bos_ids + __token_ids + __eos_ids

    def get_special_tokens_mask(self, token_ids_0: list, token_ids_1: list=None, already_has_special_tokens: bool=False) -> list:
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0,
                token_ids_1=token_ids_1,
                already_has_special_tokens=already_has_special_tokens)
        # mask matching `build_inputs_with_special_tokens`
        if token_ids_1:
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
        # default
        return [1] + ([0] * len(token_ids_0)) + [1]

    def create_token_type_ids_from_sequences(self, token_ids_0: list, token_ids_1: list=None) -> list:
        if token_ids_1:
            return (len(token_ids_0 + token_ids_1) + 3) * [0] # count the special tokens
        return (len(token_ids_0) + 2) * [0]
