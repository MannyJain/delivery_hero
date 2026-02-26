from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
import pandas as pd
from google import genai

# Ensure local `src` imports resolve when launched via `streamlit run app/streamlit_app.py`.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import get_settings
from src.data_loader import load_restaurants
from src.nlp import parse_query
from src.retriever import retrieve

st.set_page_config(page_title="AI Food Recommender", layout="centered")
st.title("🍽️ AI Food Recommender (Gemini + ChromaDB)")

settings = get_settings()
gemini_client = genai.Client(api_key=settings.google_api_key)

@st.cache_data
def restaurants_df() -> pd.DataFrame:
    return load_restaurants("data/restaurants.json")

rdf = restaurants_df()

known_restaurants = sorted(rdf["restaurant_name"].dropna().astype(str).unique().tolist())
known_locations = sorted(rdf["location"].dropna().astype(str).unique().tolist())
known_cuisines = sorted(rdf["cuisine_type"].dropna().astype(str).unique().tolist())

st.caption("Run `python scripts/build_index.py` once before using the app (it creates the local ChromaDB index).")

with st.expander("✅ Example queries", expanded=True):
    st.code("I want a burger in 30 mins")
    st.code("Order something spicy veg under 250 near Mumbai")
    st.code("From Italian Delight 13 give me a mild pasta under 400")

user_query = st.text_input("What do you want to eat?", placeholder="e.g. I want a burger in 30 mins")

c1, c2 = st.columns(2)
with c1:
    location_override = st.selectbox("Your city (optional)", ["(auto from query)"] + known_locations, index=0)
with c2:
    top_k = st.slider("How many options?", 1, 10, 5)

explain = st.toggle("Generate short LLM explanation", value=True)

if st.button("Find best items"):
    if not user_query.strip():
        st.warning("Please enter a query.")
        st.stop()

    with st.spinner("Parsing your query with Gemini..."):
        parsed = parse_query(
            client=gemini_client,
            model=settings.gemini_model,
            query=user_query.strip(),
            known_restaurants=known_restaurants,
            known_locations=known_locations,
            known_cuisines=known_cuisines,
        )

    st.subheader("🔎 Rewritten query")
    st.write(parsed.rewritten_query)

    st.subheader("🧩 Extracted filters")
    st.json(parsed.filters)

    loc = None if location_override == "(auto from query)" else location_override

    with st.spinner("Searching and ranking items..."):
        recs = retrieve(
            chroma_dir=settings.chroma_dir,
            collection_name=settings.collection_name,
            query_text=parsed.rewritten_query,
            filters=parsed.filters,
            top_k=top_k,
            candidate_k=max(30, top_k * 8),
            user_location=loc,
        )

    if not recs:
        st.error("No recommendations found. Try relaxing constraints or rebuilding the index.")
        st.stop()

    st.subheader("🍽️ Recommendations")
    for r in recs:
        st.markdown(
            f"""
**{r.item_name}**  
Restaurant: **{r.restaurant_name}** ({r.cuisine_type})  
City: {r.location} • ETA: **{r.delivery_time_minutes} min**  
Price: **₹{int(r.price)}** • Rating: {r.average_rating} • Popularity: {r.popularity_score}  
Veg: {"Yes" if r.veg else "No"} • Spice: {r.spice_level}  
Similarity: {r.similarity:.3f} • Final score: {r.final_score:.3f}  
Tags: {", ".join(r.reason_tags) if r.reason_tags else "-"}
---
"""
        )

    if explain:
        with st.spinner("Generating explanation..."):
            formatted = "\n".join(
                [f"- {r.item_name} from {r.restaurant_name} (₹{int(r.price)}, ETA {r.delivery_time_minutes}m)" for r in recs]
            )
            prompt = f"""User request:
{user_query}

Top recommended items:
{formatted}

In 2-4 sentences, explain why these items match the request. Keep it concise.
"""
            resp = gemini_client.models.generate_content(model=settings.gemini_model, contents=prompt)
            st.subheader("🤖 Explanation")
            st.write((resp.text or "").strip())
