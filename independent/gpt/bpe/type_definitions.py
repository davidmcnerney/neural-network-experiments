from typing import Dict, Tuple


VocabularyByToken = Dict[str, int]          # token's Unicode byte representation -> token index integer
VocabularyByIndex = Dict[int, str]          # token index integer -> token's Unicode byte representation
MergeList = Dict[Tuple[str, str], str]      # two token Unicode byte representations -> their merged token
