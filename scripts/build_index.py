from __future__ import annotations

from pathlib import Path

from src.config import get_settings
from src.data_loader import load_joined_dataset
from src.indexer import build_docs_from_df, rebuild_collection

def main() -> None:
    settings = get_settings()

    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"

    ## loading and joining the datasets .....

    restaurants_path = data_dir / "restaurants.json"
    menu_path = data_dir / "menu.json"

    if not restaurants_path.exists() or not menu_path.exists():
        raise FileNotFoundError(
            "Missing data files. Expected data/restaurants.json and data/menu.json."
        )

    df = load_joined_dataset(restaurants_path, menu_path)

    ### Build the “document text” (semantic searchable string)
    docs = build_docs_from_df(df)

    total = rebuild_collection(
        persist_dir=str(root / settings.chroma_dir),
        collection_name=settings.collection_name,
        docs=docs,
    )

    print(f"✅ Indexed {total} menu items into ChromaDB at: {root / settings.chroma_dir}")
    print(f"   Collection: {settings.collection_name}")

if __name__ == "__main__":
    main()
