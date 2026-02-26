from __future__ import annotations

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class Settings:
    google_api_key: str
    chroma_dir: str
    gemini_model: str
    collection_name: str = "menu_items_v1"

def get_settings() -> Settings:
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing GOOGLE_API_KEY. Put it in .env (see .env.example).")

    chroma_dir = os.getenv("CHROMA_DIR", "chroma").strip() or "chroma"
    gemini_model = os.getenv("GEMINI_MODEL", "models/gemini-2.5-flash").strip() or "models/gemini-2.5-flash"

    return Settings(
        google_api_key=api_key,
        chroma_dir=chroma_dir,
        gemini_model=gemini_model,
    )
