# app/embedding.py
import os
from sentence_transformers import SentenceTransformer
from typing import List

# Initialize the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_texts(texts: List[str]):
    """
    Generate embeddings for a list of texts using sentence-transformers
    Args:
        texts: List of strings
    Returns:
        List of embedding vectors
    """
    embeddings = model.encode(texts, convert_to_numpy=True)
    return embeddings.tolist()
