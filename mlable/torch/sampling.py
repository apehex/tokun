import torch

# NGRAMS ######################################################################

def _next(model: torch.nn.Module, ngram: list) -> int:
    __logits = model(torch.tensor([ngram]), training=False)
    __probs = torch.nn.functional.softmax(__logits, dim=-1)
    return torch.multinomial(__probs, num_samples=1).item()

def sample(model: torch.nn.Module, context: int, length: int) -> str:
    __result = []
    __ngram = context * [0]
    for __i in range(length):
        __n = _next(model=model, ngram=__ngram)
        __result.append(__n)
        __ngram = __ngram[1:] + [__n]
    return __result
