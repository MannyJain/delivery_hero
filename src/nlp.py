from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from google import genai

@dataclass
class ParsedQuery:
    filters: dict[str, Any]

_SPICE = {"mild", "medium", "spicy"}

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
    filters = data.get("filters") or {}

    if filters.get("spice_level") not in {"mild", "medium", "spicy"}:
        filters["spice_level"] = None

    for k in ["max_price", "max_delivery_time_minutes"]:
        if filters.get(k) is not None:
            try:
                filters[k] = int(filters[k])
            except Exception:
                filters[k] = None

    return ParsedQuery(filters=filters)

def parse_query(
    *,
    client: genai.Client,
    model: str,
    query: str,
    known_restaurants: list[str],
    known_locations: list[str],
    known_cuisines: list[str],
) -> ParsedQuery:
    return rewrite_and_extract_with_gemini(
            client=client,
            model=model,
            query=query,
            known_restaurants=known_restaurants,
            known_locations=known_locations,
            known_cuisines=known_cuisines,
        )
