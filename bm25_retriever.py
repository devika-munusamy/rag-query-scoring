from rank_bm25 import BM25Okapi
import numpy as np

# Common words that carry no meaning for keyword search
STOPWORDS = {
    "how", "does", "the", "a", "an", "is", "what", "where",
    "why", "do", "work", "are", "was", "were", "it", "in",
    "on", "at", "to", "for", "of", "and", "or", "with"
}


def build_bm25(documents):
    """
    Build a BM25 index from a list of documents.
    Tokenizes by lowercasing and splitting on whitespace.
    """
    tokenized = [doc.lower().split() for doc in documents]
    return BM25Okapi(tokenized)


def bm25_score(bm25_index, query):
    """
    Score all documents against a query using BM25.

    Strips stopwords so BM25 focuses on meaningful terms.
    e.g. "how does login work?" → ["login"]

    Note: if NO query word matches any document, all scores = 0.0
    That is expected BM25 behaviour — not a bug.
    Use hybrid.py to combine with cosine so results are never empty.

    Best for: Exact term search — service names, error codes, IDs.
    """
    tokens = [t for t in query.lower().split() if t not in STOPWORDS]

    # fallback: if all words were stopwords, use full query
    if not tokens:
        tokens = query.lower().split()

    scores = bm25_index.get_scores(tokens)
    return np.array(scores)


def get_top_k(scores, documents, k=3):
    top_idx = np.argsort(scores)[::-1][:k]
    return [(documents[i], float(scores[i])) for i in top_idx]


if __name__ == "__main__":
    from embedder import load_documents

    docs  = load_documents()
    bm25  = build_bm25(docs)

    queries = [
        "how does login work?",
        "PostgreSQL database connection",
        "Redis cache",
        "anomaly detection logs",
    ]

    for query in queries:
        scores  = bm25_score(bm25, query)
        results = get_top_k(scores, docs)
        print(f'\nQuery: "{query}"')
        print("── BM25 ──")
        for doc, score in results:
            print(f"  [{score:.3f}]  {doc}")