# app/utils.py
import re
from typing import List, Union

def simple_tokenize_sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]


def best_sentences_for_query(texts: Union[str, List[str]], query: str, k: int = 2) -> List[str]:
    """
    FIXED VERSION:
    - Accepts either a single string or a list of document strings.
    - Merges them safely.
    - Extracts best matching sentences.
    """

    # 1️⃣ Convert list → single string
    if isinstance(texts, list):
        texts = "\n".join([t for t in texts if isinstance(t, str)])
    elif not isinstance(texts, str):
        return []  # unsupported format

    # 2️⃣ Tokenize into sentences
    sents = simple_tokenize_sentences(texts)

    # 3️⃣ Prepare query tokens
    qtok = set(re.findall(r'\w+', query.lower()))

    scored = []
    for s in sents:
        stok = set(re.findall(r'\w+', s.lower()))
        score = len(qtok & stok)
        scored.append((score, s))

    # 4️⃣ Sort by score, highest first
    scored.sort(key=lambda x: x[0], reverse=True)

    # 5️⃣ Return best k sentences
    return [s for score, s in scored[:k]]
