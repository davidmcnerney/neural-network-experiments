from typing import Optional

import torch

from gpt.bpe import tokenizer
from gpt.bpe import type_definitions
from gpt.bpe.helpers import invert_dictionary
from gpt.model.gpt import GPT


def generate(
        model: GPT,
        vocabulary_by_token: type_definitions.VocabularyByToken,
        merge_list: type_definitions.MergeList,
        prompt: str,
        max_output_tokens: int,
        device: torch.device,
        top_p: Optional[float] = 1.0,
) -> str:
    prompt_tokens = tokenizer.tokenize(prompt, vocabulary_by_token, merge_list)
    x = torch.tensor(prompt_tokens, device=device).unsqueeze(0)
    assert x.shape == torch.Size([1, len(prompt_tokens)])
    y = model.generate(x, max_output_tokens, top_p=top_p)
    completion_tokens = y.squeeze(0).tolist()
    vocabulary_by_index = invert_dictionary(vocabulary_by_token)
    completion = tokenizer.detokenize(completion_tokens, vocabulary_by_index)
    return completion
