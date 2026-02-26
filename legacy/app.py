import json
import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from google import genai

# =====================================================
# 1️⃣ Streamlit Config
# =====================================================

st.set_page_config(page_title="AI Food Recommender", layout="centered")
st.title("🍽 AI Food Recommendation System")

# =====================================================
# 2️⃣ Load Gemini
# =====================================================

with open("config.json", "r") as f:
    config = json.load(f)

gemini = genai.Client(api_key=config["GOOGLE_API_KEY"])
MODEL_NAME = "models/gemini-2.5-flash"

# =====================================================
# 3️⃣ Load Embedding Model (Cached)
# =====================================================

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedding_model = load_embedding_model()

# =====================================================
# 4️⃣ Load Dataset (Cached)
# =====================================================

@st.cache_data
def load_data():
    restaurants_df = pd.read_json("restaurants.json")
    menu_df = pd.read_json("menu.json")
    df = menu_df.merge(restaurants_df, on="restaurant_id", how="left")
    df = pd.read_pickle("menu_with_embeddings_local.pkl")
    return df

df = load_data()

# =====================================================
# 5️⃣ Gemini: Rewrite + Extract (Single Call)
# =====================================================

def rewrite_and_extract(query):

    prompt = f"""
You are an intelligent food assistant.

From the user query:

"{query}"

1. Rewrite it into a clear descriptive search sentence.
2. Extract structured filters.

Return ONLY valid JSON in this format:

{{
  "rewritten_query": "...",
  "filters": {{
      "veg": true/false/null,
      "spice_level": "mild/medium/spicy/null",
      "max_price": number/null
  }}
}}

Do not explain anything.
Return JSON only.
"""

    response = gemini.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    try:
        result = json.loads(response.text.strip())
        return result["rewritten_query"], result["filters"]
    except:
        return query, {}

# =====================================================
# 6️⃣ Gemini Explanation Layer
# =====================================================

def generate_explanation(user_query, recommendations):

    formatted = "\n".join(
        [
            f"{row['item_name']} from {row['restaurant_name']} "
            f"(₹{row['price']}, Rating: {row['average_rating']})"
            for _, row in recommendations.iterrows()
        ]
    )

    prompt = f"""
User request:
{user_query}

Recommended dishes:
{formatted}

Explain briefly why these match the user's request.
"""

    response = gemini.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    return response.text.strip()

# =====================================================
# 7️⃣ Recommendation Logic
# =====================================================

def recommend(query, top_k=5):

    rewritten_query, filters = rewrite_and_extract(query)

    filtered_df = df.copy()

    # Structured filtering
    if filters.get("veg") is not None:
        filtered_df = filtered_df[filtered_df["veg"] == filters["veg"]]

    if filters.get("spice_level"):
        filtered_df = filtered_df[
            filtered_df["spice_level"] == filters["spice_level"]
        ]

    if filters.get("max_price"):
        filtered_df = filtered_df[
            filtered_df["price"] <= filters["max_price"]
        ]

    if len(filtered_df) == 0:
        filtered_df = df.copy()

    # Semantic similarity ranking
    query_embedding = embedding_model.encode([rewritten_query])[0]
    item_embeddings = np.vstack(filtered_df["embedding"].values)

    similarities = cosine_similarity(
        [query_embedding],
        item_embeddings
    )[0]

    filtered_df = filtered_df.copy()
    filtered_df["similarity"] = similarities

    ranked_df = filtered_df.sort_values(
        by="similarity",
        ascending=False
    ).head(top_k)

    explanation = generate_explanation(query, ranked_df)

    return ranked_df, explanation, rewritten_query, filters

# =====================================================
# 8️⃣ UI
# =====================================================

user_query = st.text_input(
    "Enter your food preference:",
    placeholder="e.g. spicy non veg under 300"
)

top_k = st.slider("Number of recommendations:", 1, 10, 5)

if st.button("Get Recommendations"):

    if user_query.strip() == "":
        st.warning("Please enter a query.")
    else:
        with st.spinner("AI is thinking..."):
            results, explanation, rewritten_query, filters = recommend(user_query, top_k)

        st.subheader("🔍 Rewritten Query")
        st.write(rewritten_query)

        st.subheader("🧠 Extracted Filters")
        st.json(filters)

        st.subheader("🍽 Top Recommendations")

        for _, row in results.iterrows():
            st.markdown(
                f"""
**{row['item_name']}**  
Restaurant: {row['restaurant_name']}  
Price: ₹{row['price']}  
Rating: {row['average_rating']}  
Spice Level: {row['spice_level']}  
Vegetarian: {"Yes" if row['veg'] else "No"}  
Similarity Score: {round(row['similarity'], 3)}
---
"""
            )

        st.subheader("🤖 AI Explanation")
        st.write(explanation)