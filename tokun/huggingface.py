import copy
import json

import transformers

# UTF-32-BE TOKENIZATION ######################################################

class ByteTokenizer(transformers.PreTrainedTokenizer):
    def __init__(self, encoding: str='utf-32-be', **kwargs) -> None:
        __kwargs = copy.deepcopy(kwargs)
        # special tokens
        __kwargs['bos_token'] = __kwargs.get('bos_token', '\u0002')
        __kwargs['eos_token'] = __kwargs.get('eos_token', '\u0003')
        __kwargs['unk_token'] = __kwargs.get('unk_token', '\u001a')
        __kwargs['sep_token'] = __kwargs.get('sep_token', '\u001d')
        __kwargs['pad_token'] = __kwargs.get('pad_token', '\u0000')
        __kwargs['cls_token'] = __kwargs.get('cls_token', '\u0011')
        # defaults
        __kwargs['split_special_tokens'] = True # 4 bytes for the special tokens too
        __kwargs['vocab_size'] = 256
        # properties
        self._vocab_size = 256
        self._encoding = encoding
        # init
        super(ByteTokenizer, self).__init__(**__kwargs)

    def _tokenize(self, text: str, **kwargs) -> list:
        return list(chr(__b) for __b in text.encode(self._encoding))

    def _convert_token_to_id(self, token: str) -> int:
        return ord(token)

    def _convert_id_to_token(self, index: int) -> str:
        return chr(index)

    def convert_tokens_to_string(self, tokens: iter) -> str:
        return bytes(ord(__c) for __c in tokens).decode(self._encoding)

    def build_inputs_with_special_tokens(self, token_ids_0: list, token_ids_1: list=None) -> list:
        __cls_ids = [self._convert_token_to_id(__t) for __t in self.tokenize(self.cls_token, split_special_tokens=True)]
        __sep_ids = [self._convert_token_to_id(__t) for __t in self.tokenize(self.sep_token, split_special_tokens=True)]
        __ids = __cls_ids + token_ids_0 + __sep_ids
        return __ids + token_ids_1 + __sep_ids if token_ids_1 else __ids

    def get_vocab(self) -> dict:
        return {chr(__i): __i for __i in range(self._vocab_size)}

    def save_vocabulary(self, save_directory: str, **kwargs) -> tuple:
        __prefix = kwargs.get('filename_prefix', '')
        __path = "{}/{}vocab.json".format(save_directory, __prefix if __prefix else '')
        with open(__path, "w") as __file:
            json.dump(self.get_vocab(), __file)
        return (__path,)

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @vocab_size.setter
    def vocab_size(self, size: int) -> None:
        self._vocab_size = size
