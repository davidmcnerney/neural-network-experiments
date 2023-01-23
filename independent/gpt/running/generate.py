import torch

from independent.gpt.bpe import tokenizer
from independent.gpt.bpe import type_definitions
from independent.gpt.bpe.helpers import invert_dictionary
from independent.gpt.model.gpt import GPT


def generate(
        model: GPT,
        vocabulary_by_token: type_definitions.VocabularyByToken,
        merge_list: type_definitions.MergeList,
        prompt: str,
        max_output_tokens: int,
        device: torch.device,
) -> str:
    prompt_tokens = tokenizer.tokenize(prompt, vocabulary_by_token, merge_list)
    x = torch.tensor(prompt_tokens, device=device).unsqueeze(0)
    assert x.shape == torch.Size([1, len(prompt_tokens)])
    y = model.generate(x, max_output_tokens)
    completion_tokens = y.squeeze(0).tolist()
    vocabulary_by_index = invert_dictionary(vocabulary_by_token)
    completion = tokenizer.detokenize(completion_tokens, vocabulary_by_index)
    return completion
