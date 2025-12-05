# app/utils.py
import re
from typing import List, Union


def simple_tokenize_sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]


def best_sentences_for_query(
    texts: Union[str, List[str]],
    query: str,
    k: int = 2
) -> List[str]:
    """
    Robust version:
    - `texts` can be a single string OR list of strings (multiple docs)
    - `query` must be a string
    - returns top-k sentences that overlap most with query words
    """

    # 🔹 Normalize texts → one big string
    if isinstance(texts, list):
        texts = "\n".join(t for t in texts if isinstance(t, str))
    elif not isinstance(texts, str):
        texts = str(texts)

    # 🔹 Ensure query is a string
    if not isinstance(query, str):
        query = str(query)

    # 1) Split into sentences
    sents = simple_tokenize_sentences(texts)

    # 2) Tokenize query
    qtok = set(re.findall(r"\w+", query.lower()))

    scored: List[tuple[int, str]] = []
    for s in sents:
        stok = set(re.findall(r"\w+", s.lower()))
        score = len(qtok & stok)
        scored.append((score, s))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [s for score, s in scored[:k]] if scored else sents[:k]
