from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from .embeddings import embed_text
from .indexer import get_chroma_client

@dataclass
class Recommendation:
    item_id: int
    restaurant_id: int
    item_name: str
    restaurant_name: str
    location: str
    cuisine_type: str
    price: float
    veg: Optional[bool]
    spice_level: str
    delivery_time_minutes: int
    average_rating: float
    popularity_score: int
    similarity: float
    final_score: float
    reason_tags: list[str]

def _normalize(x: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return float((x - lo) / (hi - lo))

def retrieve(
    *,
    chroma_dir: str,
    collection_name: str,
    query_text: str,
    filters: dict[str, Any],
    top_k: int = 5,
    candidate_k: int = 20,
    user_location: Optional[str] = None,
) -> list[Recommendation]:
    client = get_chroma_client(chroma_dir)
    col = client.get_or_create_collection(name=collection_name)
    
    ## after parsing the query, we need to embed the query text to get the nearest neighbor ....
    q_emb = embed_text(query_text)
    
    ## querying the vector db ....
    ## for now top 20 candidates are returned ...
    res = col.query(
        query_embeddings=[q_emb],
        n_results=candidate_k,
        include=["metadatas", "distances"],
    )
    
    ## now we have the semantic distances for neighbors
    ## and their metadata for further ranking .....
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]
    if not metas:
        return []

    sims = [float(1.0 - d) for d in dists]  # cosine distance -> similarity
    

    ## now from the filtered candidate metadata, we need to extract the delivery time, rating and popularity ...
    delivery = [int(m.get("delivery_time_minutes", 999)) for m in metas]
    rating = [float(m.get("average_rating", 0.0)) for m in metas]
    popularity = [int(m.get("popularity_score", 0)) for m in metas]

    dmin, dmax = min(delivery), max(delivery)
    rmin, rmax = min(rating), max(rating)
    pmin, pmax = min(popularity), max(popularity)
    
    ## filters are the constraints that the user has mentioned in the query ...
    max_price = filters.get("max_price")
    max_dt = filters.get("max_delivery_time_minutes")
    spice = filters.get("spice_level")
    veg = filters.get("veg")
    rest_name = (filters.get("restaurant_name") or "").strip().lower() or None
    loc = (filters.get("location") or "").strip().lower() or None
    cuisine = (filters.get("cuisine_type") or "").strip().lower() or None
    
    ## preparing the output list of recommendations ...
    out: list[Recommendation] = []

    for m, sim in zip(metas, sims):
        ## tags are the reason tags for the recommendation ...
        tags: list[str] = []

        if rest_name and str(m.get("restaurant_name", "")).lower() != rest_name:
            continue
        
        ## if the user has mentioned a location in the query, then we need to check if the restaurant is in that location ...
        effective_loc = user_location or loc

        if effective_loc:
            if str(m.get("location", "")).lower() != effective_loc.lower():
                continue
            tags.append("location_match")

        if veg is not None:
            if m.get("veg") is not None and bool(m.get("veg")) != bool(veg):
                continue
            tags.append("veg_match")

        if spice:
            if str(m.get("spice_level", "")).lower() != spice.lower():
                continue
            tags.append("spice_match")

        if cuisine:
            if str(m.get("cuisine_type", "")).lower() != cuisine.lower():
                continue
            tags.append("cuisine_match")
        
        ## if some filters are not matching lets say price 
        ## calculating some sort of penalty scores to rank them ...
        price = float(m.get("price", 1e9))
        budget_penalty = 0.0
        if max_price is not None and price > float(max_price):
            tags.append("over_budget")
            over_ratio = min(1.0, (price - float(max_price)) / max(float(max_price), 1.0))
            budget_penalty = 0.25 * over_ratio
        elif max_price is not None:
            tags.append("within_budget")

        dt = int(m.get("delivery_time_minutes", 999))
        if max_dt is not None and dt > int(max_dt):
            continue
        if max_dt is not None:
            tags.append("within_time")

        dt_score = 1.0 - _normalize(dt, dmin, dmax)
        rating_score = _normalize(float(m.get("average_rating", 0.0)), rmin, rmax)
        pop_score = _normalize(float(m.get("popularity_score", 0)), pmin, pmax)

        final = (
            0.70 * float(sim) +
            0.15 * dt_score +
            0.10 * rating_score +
            0.05 * pop_score
        ) - budget_penalty

        out.append(
            Recommendation(
                item_id=int(m.get("item_id")),
                restaurant_id=int(m.get("restaurant_id")),
                item_name=str(m.get("item_name")),
                restaurant_name=str(m.get("restaurant_name")),
                location=str(m.get("location")),
                cuisine_type=str(m.get("cuisine_type")),
                price=price,
                veg=m.get("veg") if m.get("veg") is None else bool(m.get("veg")),
                spice_level=str(m.get("spice_level")),
                delivery_time_minutes=dt,
                average_rating=float(m.get("average_rating", 0.0)),
                popularity_score=int(m.get("popularity_score", 0)),
                similarity=float(sim),
                final_score=float(final),
                reason_tags=tags,
            )
        )
    
    ## sorting the recommendations based on the final score ...
    ## higher the final score, higher the recommendation ...
    ## final score is a hybrid score of semantic similarity, delivery time, rating and popularity ...
    ## and some sort of penalty scores for over budget and over time ...
    out.sort(key=lambda r: r.final_score, reverse=True)
    return out[:top_k]
