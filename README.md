# rag-query-scoring

A clean, runnable comparison of RAG query scoring methods — built to go alongside the Medium blog.

## What's Inside

5 scoring methods, all running on the same query so you can see exactly how they differ:

| Method | File | Speed | Best For |
|---|---|---|---|
| Cosine Similarity | `retriever.py` | Fast | General RAG |
| Dot Product | `retriever.py` | Fastest | Large-scale / FAISS |
| BM25 | `bm25_retriever.py` | Fast | Exact keyword search |
| Hybrid (BM25 + Cosine) | `hybrid.py` | Medium | Production systems |
| Cross-Encoder Reranker | `reranker.py` | Slow | High-quality Q&A |

## Folder Structure

```
rag-query-scoring/
├── main.py               ← Run all methods, compare results
├── embedder.py           ← Load any embedding model
├── retriever.py          ← Cosine + dot product scoring
├── bm25_retriever.py     ← BM25 keyword scoring
├── hybrid.py             ← BM25 + cosine combined
├── reranker.py           ← Cross-encoder reranking
├── pipeline.py           ← Full end-to-end RAG pipeline
├── documents/
│   └── sample_docs.txt   ← Knowledge base (add your own docs here)
└── requirements.txt
```

## Setup

```bash
git clone https://github.com/devika-munusamy/rag-query-scoring
cd rag-query-scoring
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
# Compare all 5 scoring methods side by side
python main.py

# Custom query
python main.py --query "what database is used?"

# Full pipeline (embed → hybrid search → rerank → prompt)
python pipeline.py
```

## Sample Output

```
Query: "how does login work?"

── 1. Cosine Similarity ──
  [0.743]  Auth service manages user login and sessions.
  [0.421]  API gateway routes requests to backend services.
  [0.318]  The payment service handles all transactions.

── 4. Hybrid (BM25 + Cosine) ──
  [0.814]  Auth service manages user login and sessions.
  [0.512]  API gateway routes requests to backend services.
  [0.301]  Order service tracks order lifecycle.

── 5. Cross-Encoder Reranker ──
  [0.923]  Auth service manages user login and sessions.
  [0.341]  API gateway routes requests to backend services.
  [0.187]  The DB service connects to PostgreSQL.
```

## Blog

Read the full explanation → [Medium post link here]

## Models Covered

- `all-MiniLM-L6-v2` — fast, free, great for prototypes
- `BAAI/bge-large-en-v1.5` — best open source, top MTEB
- `OpenAI text-embedding-3-small` — best API option
- `BAAI/bge-m3` — multilingual, 100+ languages
- `Cohere embed-v4` — enterprise, highest accuracy
