from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import chromadb
from chromadb.config import Settings as ChromaSettings

from .embeddings import embed_texts

@dataclass(frozen=True)
class IndexDoc:
    doc_id: str
    text: str
    metadata: dict

def _safe_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default

def _safe_int(x, default=0) -> int:
    try:
        return int(x)
    except Exception:
        return default

def build_docs_from_df(df) -> list[IndexDoc]:
    docs: list[IndexDoc] = []
    for _, row in df.iterrows():
        doc_id = f"item_{int(row['item_id'])}"

        text = (
            f"{row.get('item_name','')}. {row.get('description','')}. "
            f"Category: {row.get('category','')}. "
            f"Cuisine: {row.get('cuisine_type','')}. "
            f"Spice: {row.get('spice_level','')}. "
            f"{'Vegetarian' if bool(row.get('veg')) else 'Non-vegetarian'}. "
            f"Restaurant: {row.get('restaurant_name','')}. "
            f"Location: {row.get('location','')}. "
        ).strip()

        metadata = {
            "item_id": _safe_int(row.get("item_id")),
            "restaurant_id": _safe_int(row.get("restaurant_id")),
            "item_name": str(row.get("item_name", "")),
            "category": str(row.get("category", "")),
            "price": _safe_float(row.get("price")),
            "veg": bool(row.get("veg")) if row.get("veg") is not None else None,
            "spice_level": str(row.get("spice_level", "")),
            "calories": _safe_int(row.get("calories", 0)),
            "is_chef_special": bool(row.get("is_chef_special")) if row.get("is_chef_special") is not None else False,
            "restaurant_name": str(row.get("restaurant_name", "")),
            "cuisine_type": str(row.get("cuisine_type", "")),
            "average_rating": _safe_float(row.get("average_rating")),
            "price_range": str(row.get("price_range", "")),
            "location": str(row.get("location", "")),
            "delivery_time_minutes": _safe_int(row.get("delivery_time_minutes", 999)),
            "is_pure_veg": bool(row.get("is_pure_veg")) if row.get("is_pure_veg") is not None else None,
            "popularity_score": _safe_int(row.get("popularity_score", 0)),
        }
        docs.append(IndexDoc(doc_id=doc_id, text=text, metadata=metadata))
    return docs

def get_chroma_client(persist_dir: str) -> chromadb.PersistentClient:
    return chromadb.PersistentClient(
        path=persist_dir,
        settings=ChromaSettings(anonymized_telemetry=False),
    )

def rebuild_collection(
    *,
    persist_dir: str,
    collection_name: str,
    docs: Iterable[IndexDoc],
    batch_size: int = 256,
) -> int:
    client = get_chroma_client(persist_dir)

    existing = [c.name for c in client.list_collections()]
    if collection_name in existing:
        client.delete_collection(collection_name)

    col = client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})

    docs_list = list(docs)
    texts = [d.text for d in docs_list]
    embeddings = embed_texts(texts)

    total = 0
    for i in range(0, len(docs_list), batch_size):
        batch_docs = docs_list[i:i+batch_size]
        batch_emb = embeddings[i:i+batch_size]
        col.add(
            ids=[d.doc_id for d in batch_docs],
            documents=[d.text for d in batch_docs],
            metadatas=[d.metadata for d in batch_docs],
            embeddings=batch_emb,
        )
        total += len(batch_docs)

    # ChromaDB >=0.5 persists automatically; older clients expose persist().
    persist_fn = getattr(client, "persist", None)
    if callable(persist_fn):
        persist_fn()
    return total
