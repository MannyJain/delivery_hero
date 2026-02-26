from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

def load_restaurants(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    return pd.DataFrame(data)

def load_menu(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    return pd.DataFrame(data)

def load_joined_dataset(restaurants_path: str | Path, menu_path: str | Path) -> pd.DataFrame:
    restaurants_df = load_restaurants(restaurants_path)
    menu_df = load_menu(menu_path)

    df = menu_df.merge(restaurants_df, on="restaurant_id", how="left", suffixes=("", "_restaurant"))

    required_cols = [
        "item_id", "restaurant_id", "item_name", "description", "category", "price", "veg", "spice_level",
        "restaurant_name", "cuisine_type", "average_rating", "price_range", "location",
        "delivery_time_minutes", "is_pure_veg", "popularity_score"
    ]
    for c in required_cols:
        if c not in df.columns:
            df[c] = None
    return df
