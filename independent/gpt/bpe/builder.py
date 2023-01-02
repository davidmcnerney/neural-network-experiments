from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from independent.gpt.bpe import text_processing


Vocabulary = Dict[int, str]
Merge = Tuple[str, str, str]   # first, second, merged (we don't really need `merged`)


def build_vocabulary_and_merge_list(
        training_text: str,
        count_merges: int,
) -> Tuple[Vocabulary, List[Merge]]:
    """
    Builds:
        - a vocabulary: a dictionary mapping token index integers to token strings
        - a merge list: a list of pairs of tokens which should be combined to yield
            another token in our vocabulary
    """

    vocabulary: Vocabulary = {}
    merge_list: List[Merge] = []
    next_available_index = 0

    # Establish initial vocabulary of the 256 individual bytes
    vocabulary.update(text_processing.BYTE_TO_UNICODE_REPRESENTATION)
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

        next_pair_frequency = pair_frequencies[-1]
        first, second, _ = next_pair_frequency
        merged = first + second

        merge_list.append((first, second, merged))
        vocabulary[next_available_index] = merged
        next_available_index += 1

        merges_remaining -= 1

    return vocabulary, merge_list


PairFrequency = Tuple[str, str, int]


def _compute_pair_frequencies(
        text: str,
        merge_list: List[Merge],
) -> List[PairFrequency]:
    """
    Tokenizes given the merge_list, finds all pairs of adjacent tokens,
    and returns a list of these pairs together with their counts, ordered by
    frequency
    """
    frequencies: Dict[Tuple[str, str], int] = defaultdict(int)
    for coarse_token in _coarse_tokenize(text):
        for first, second in _pairs(_fine_tokenize(coarse_token, merge_list)):
            frequencies[first, second] += 1
    tuples = [(pair[0], pair[1], count) for pair, count in frequencies.items()]
    return sorted(tuples, key=lambda t: t[2])


def _coarse_tokenize(text: str) -> List[str]:
    return [
        text_processing.to_unicode_bytes(token)
        for token in text_processing.pretokenize(text)
    ]


def _fine_tokenize(string: str, merge_list: List[Merge]) -> List[str]:
    # Start by breaking the string into 1 byte tokens
    tokens = list(string)

    # apply all possible merges in the merge list
    # TODO: this is gonna be slow with the typical length of merge list
    #       - consider making merge list an ordered dict so that we can check presence fast
    for merge in merge_list:
        tokens = _apply_merge(tokens, merge)

    return tokens


def _apply_merge(in_tokens: List[str], merge: Merge) -> List[str]:
    first, second, merged = merge

    out_tokens: List[str] = []
    startIndex = 0
    while startIndex < len(in_tokens):
        if startIndex < len(in_tokens) - 1 and in_tokens[startIndex] == first and in_tokens[startIndex + 1] == second:
            out_tokens.append(merged)
            startIndex += 2
        else:
            out_tokens.append(in_tokens[startIndex])
            startIndex += 1
    return out_tokens


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


def _insert_pair_frequency(
        pair_frequencies: List[PairFrequency],
        new_pair_frequency: PairFrequency,
) -> None:
    index_to_insert_at: Optional[int] = None
    for index in range(len(pair_frequencies)):
        if pair_frequencies[index][2] > new_pair_frequency[2]:
            index_to_insert_at = index
            break
    if index_to_insert_at is not None:
        pair_frequencies.insert(index_to_insert_at, new_pair_frequency)
    else:
        pair_frequencies.append(new_pair_frequency)


def remove_base_byte_vocab(vocab: Vocabulary) -> Vocabulary:
    copy = dict(vocab)
    for index in range(256):
        del copy[index]
    return copy


def summarize_vocab(vocab: Vocabulary) -> None:
    for index, string in vocab.items():
        print(f"{index:6}: {string}")


def summarize_merges(merges: List[Merge]) -> None:
    for first, second, merged in merges:
        print(f"{first} + {second} -> {merged}")


def _output_progress_dot() -> None:
    print(".", end="", flush=True)
