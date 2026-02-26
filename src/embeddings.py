from __future__ import annotations

from functools import lru_cache
from sentence_transformers import SentenceTransformer

@lru_cache(maxsize=1)
def get_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    return SentenceTransformer(model_name)

def embed_texts(texts: list[str]) -> list[list[float]]:
    model = get_embedding_model()
    emb = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    return emb.tolist()

def embed_text(text: str) -> list[float]:
    return embed_texts([text])[0]
