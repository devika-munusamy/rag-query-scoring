import numpy as np
from retriever import cosine_score
from bm25_retriever import bm25_score


def hybrid_score(query, query_emb, doc_embs, bm25_index, alpha=0.4):
    """
    Combine BM25 (keyword) and cosine (semantic) into a single score.

    alpha controls the BM25 weight:
        alpha=0.0  → pure cosine (semantic only)
        alpha=0.5  → equal weight
        alpha=1.0  → pure BM25  (keyword only)

    Default alpha=0.4 gives slightly more weight to semantic similarity.

    Best for: Production systems. Catches both exact terms and semantic meaning.
    """
    cos_scores  = cosine_score(query_emb, doc_embs)
    bm25_scores = bm25_score(bm25_index, query)

    # Normalize both to [0, 1] range before combining
    cos_norm  = cos_scores  / (cos_scores.max()  + 1e-9)
    bm25_norm = bm25_scores / (bm25_scores.max() + 1e-9)

    combined = alpha * bm25_norm + (1 - alpha) * cos_norm
    return combined


def get_top_k(scores, documents, k=3):
    """
    Return top-k (document, score) pairs sorted by score descending.
    """
    top_idx = np.argsort(scores)[::-1][:k]
    return [(documents[i], float(scores[i])) for i in top_idx]


if __name__ == "__main__":
    from embedder import load_model, embed_documents, embed_query, load_documents
    from bm25_retriever import build_bm25

    docs      = load_documents()
    model     = load_model("minilm")
    doc_embs  = embed_documents(model, docs)
    bm25      = build_bm25(docs)
    query     = "how does login work?"
    query_emb = embed_query(model, query)

    scores  = hybrid_score(query, query_emb, doc_embs, bm25, alpha=0.4)
    results = get_top_k(scores, docs)

    print(f'\nQuery: "{query}"\n')
    print("── Hybrid Score (BM25 + Cosine, alpha=0.4) ──")
    for doc, score in results:
        print(f"  [{score:.3f}]  {doc}")

    # Try different alpha values
    print("\n── Alpha comparison ──")
    for a in [0.0, 0.3, 0.5, 0.7, 1.0]:
        s       = hybrid_score(query, query_emb, doc_embs, bm25, alpha=a)
        top_doc = docs[np.argmax(s)]
        print(f"  alpha={a}  →  top: {top_doc[:50]}")