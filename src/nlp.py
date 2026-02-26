from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from google import genai

@dataclass
class ParsedQuery:
    rewritten_query: str
    filters: dict[str, Any]

_SPICE = {"mild", "medium", "spicy"}

def _heuristic_parse(query: str) -> ParsedQuery:
    q = query.lower()

    filters: dict[str, Any] = {
        "veg": None,
        "spice_level": None,
        "max_price": None,
        "max_delivery_time_minutes": None,
        "restaurant_name": None,
        "location": None,
        "cuisine_type": None,
    }

    if re.search(r"\b(veg|vegetarian|pure veg)\b", q):
        filters["veg"] = True
    if re.search(r"\b(non\s*veg|non-veg|chicken|mutton|fish|steak)\b", q):
        if filters["veg"] is None:
            filters["veg"] = False

    for s in _SPICE:
        if re.search(rf"\b{s}\b", q):
            filters["spice_level"] = s
            break

    m = re.search(r"(under|below|max)\s*₹?\s*(\d+)", q)
    if m:
        filters["max_price"] = int(m.group(2))

    m = re.search(r"(in|within)\s*(\d+)\s*(min|mins|minutes)", q)
    if m:
        filters["max_delivery_time_minutes"] = int(m.group(2))

    rewritten = re.sub(r"(under|below|max)\s*₹?\s*\d+", "", query, flags=re.I)
    rewritten = re.sub(r"(in|within)\s*\d+\s*(min|mins|minutes)", "", rewritten, flags=re.I)
    rewritten = rewritten.strip()

    return ParsedQuery(rewritten_query=rewritten or query, filters=filters)

def rewrite_and_extract_with_gemini(
    *,
    client: genai.Client,
    model: str,
    query: str,
    known_restaurants: list[str],
    known_locations: list[str],
    known_cuisines: list[str],
) -> ParsedQuery:
    prompt = f"""
You are an intelligent food assistant.

User query:
\"\"\"{query}\"\"\"

Return ONLY valid JSON in this exact schema:

{{
  "rewritten_query": "string",
  "filters": {{
    "veg": true|false|null,
    "spice_level": "mild"|"medium"|"spicy"|null,
    "max_price": number|null,
    "max_delivery_time_minutes": number|null,
    "restaurant_name": string|null,
    "location": string|null,
    "cuisine_type": string|null
  }}
}}

Extraction guidance:
- If the user mentions a delivery time like "in 30 mins" set max_delivery_time_minutes=30.
- If the user explicitly mentions a restaurant, set restaurant_name EXACTLY as one of the known restaurants (or null).
- If the user mentions a city/location, set location EXACTLY as one of known locations (or null).
- If cuisine is implied (e.g., "Italian"), set cuisine_type EXACTLY as one of known cuisines (or null).

Known restaurants (choose exact match or null):
{json.dumps(known_restaurants, ensure_ascii=False)}

Known locations (choose exact match or null):
{json.dumps(known_locations, ensure_ascii=False)}

Known cuisines (choose exact match or null):
{json.dumps(known_cuisines, ensure_ascii=False)}

Return JSON only. No markdown. No explanation.
""".strip()

    resp = client.models.generate_content(model=model, contents=prompt)
    text = (resp.text or "").strip()

    data = json.loads(text)
    rewritten = str(data.get("rewritten_query") or query)
    filters = data.get("filters") or {}

    if filters.get("spice_level") not in {"mild", "medium", "spicy"}:
        filters["spice_level"] = None

    for k in ["max_price", "max_delivery_time_minutes"]:
        if filters.get(k) is not None:
            try:
                filters[k] = int(filters[k])
            except Exception:
                filters[k] = None

    return ParsedQuery(rewritten_query=rewritten, filters=filters)

def parse_query(
    *,
    client: genai.Client,
    model: str,
    query: str,
    known_restaurants: list[str],
    known_locations: list[str],
    known_cuisines: list[str],
) -> ParsedQuery:
    try:
        return rewrite_and_extract_with_gemini(
            client=client,
            model=model,
            query=query,
            known_restaurants=known_restaurants,
            known_locations=known_locations,
            known_cuisines=known_cuisines,
        )
    except Exception:
        return _heuristic_parse(query)
