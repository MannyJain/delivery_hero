# LLM Food Recommender (Gemini + ChromaDB + Streamlit)

End-to-end demo:
- Indexes `data/restaurants.json` + `data/menu.json` into **ChromaDB** (persistent, local).
- Uses embeddings for approximate matching (semantic search).
- Uses **Gemini** to (1) rewrite the query + extract structured filters and (2) optionally produce a short explanation.

## Folder structure

```
llm-food-recommender/
  data/
    restaurants.json
    menu.json
  chroma/                 # created at runtime (persistent vector store)
  scripts/
    build_index.py        # builds/refreshes Chroma collection from data/*.json
  src/
    config.py
    data_loader.py
    embeddings.py
    indexer.py
    nlp.py
    retriever.py
  app/
    streamlit_app.py      # Streamlit UI
  requirements.txt
  .env.example
```

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # put GOOGLE_API_KEY inside
python scripts/build_index.py
streamlit run app/streamlit_app.py
```

## Sample queries
1) `I want a burger in 30 mins`  
2) `Order something spicy veg under 250 near Mumbai`  
3) `From Italian Delight 13 give me a mild pasta under 400`  

> Your synthetic dataset may not contain the exact word "burger".
> Semantic search will still return approximate matches.

## Ranking
Chroma returns semantic candidates, then we re-rank with a hybrid score:
- semantic similarity (dominant)
- faster delivery boost
- rating & popularity boosts
- over-budget soft penalty
