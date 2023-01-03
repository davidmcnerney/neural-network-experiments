from collections import defaultdict
from typing import Dict, List, Tuple

from independent.gpt.bpe import text_processing
from independent.gpt.bpe import tokenizer
from independent.gpt.bpe import type_definitions


#
# Builder
#

def build_vocabulary_and_merge_list(
        training_text: str,
        count_merges: int,
) -> Tuple[type_definitions.VocabularyByToken, type_definitions.MergeList]:
    """
    Builds:
        - a vocabulary: a dictionary mapping token strings to token index integers
        - a merge list: a list of pairs of tokens which should be combined to yield
            another token in our vocabulary
    """

    vocabulary: type_definitions.VocabularyByToken = {}
    merge_list: type_definitions.MergeList = {}
    next_available_index = 0

    # Establish initial vocabulary of the 256 individual bytes
    vocabulary.update(text_processing.UNICODE_REPRESENTATION_TO_BYTE)
    next_available_index += 256

    # Iteratively add merges until either we've done the allowed count_merges,
    # or there is nothing left to merge
    merges_remaining = count_merges
    while merges_remaining > 0:
        _output_progress_dot()

        pair_frequencies = _compute_pair_frequencies(training_text, merge_list)

        # Should not normally happen
        if len(pair_frequencies) == 0:
            break

        highest_frequency = pair_frequencies[-1]
        first, second, _ = highest_frequency
        merged = first + second

        merge_list[first, second] = merged
        vocabulary[merged] = next_available_index
        next_available_index += 1

        merges_remaining -= 1

    return vocabulary, merge_list


PairFrequency = Tuple[str, str, int]


def _compute_pair_frequencies(
        text: str,
        merge_list: type_definitions.MergeList,
) -> List[PairFrequency]:
    """
    Tokenizes given the merge_list, finds all pairs of adjacent tokens,
    and returns a list of these pairs together with their counts, ordered by
    frequency
    """
    frequencies: Dict[Tuple[str, str], int] = defaultdict(int)
    for coarse_token in tokenizer.coarse_tokenize(text):
        for first, second in _pairs(tokenizer.fine_tokenize(coarse_token, merge_list)):
            frequencies[first, second] += 1
    tuples = [(pair[0], pair[1], count) for pair, count in frequencies.items()]
    return sorted(tuples, key=lambda t: t[2])


def _pairs(
        tokens: List[str],
) -> List[Tuple[str, str]]:
    """
    Returns the possible token pairs in a list, including duplicates.
    For example:
        ["H", "e", "l", "l", "o"] -> [("H", "e"), ("e", "l"), ("l", "l"), ("l", "o")]
    """
    if len(tokens) < 2:
        return []
    pairs: List[Tuple[str, str]] = []
    first = tokens[0]
    for second in tokens[1:]:
        pairs.append((first, second))
        first = second
    return pairs


def _output_progress_dot() -> None:
    print(".", end="", flush=True)
