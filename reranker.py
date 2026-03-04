from sentence_transformers import CrossEncoder
import numpy as np


RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def load_reranker():
    """
    Load the cross-encoder reranker model.
    This model reads query + document together for more accurate scoring.
    Slower than embeddings but significantly more accurate.
    """
    print(f"Loading reranker: {RERANKER_MODEL}")
    return CrossEncoder(RERANKER_MODEL)


def rerank(reranker, query, candidate_docs):
    """
    Score each (query, document) pair using the cross-encoder.

    Unlike cosine similarity which embeds query and document separately,
    cross-encoders see both at once — giving much more accurate relevance scores.

    Best for: Second-pass reranking after fast cosine/BM25 retrieval.
    Use it on top-5 or top-10 candidates, not the full document set.

    Returns: list of (document, score) sorted by score descending.
    """
    pairs  = [[query, doc] for doc in candidate_docs]
    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(candidate_docs, scores.tolist()),
        key=lambda x: x[1],
        reverse=True
    )
    return ranked


if __name__ == "__main__":
    from embedder import load_model, embed_documents, embed_query, load_documents
    from retriever import cosine_score

    docs      = load_documents()
    model     = load_model("minilm")
    doc_embs  = embed_documents(model, docs)
    query     = "how does login work?"
    query_emb = embed_query(model, query)

    # Stage 1: fast cosine retrieval — get top 5 candidates
    cos_scores = cosine_score(query_emb, doc_embs)
    top5_idx   = np.argsort(cos_scores)[::-1][:5]
    top5_docs  = [docs[i] for i in top5_idx]

    print(f'\nQuery: "{query}"\n')
    print("── Stage 1: Cosine top-5 ──")
    for i in top5_idx:
        print(f"  [{cos_scores[i]:.3f}]  {docs[i]}")

    # Stage 2: cross-encoder reranking
    reranker = load_reranker()
    ranked   = rerank(reranker, query, top5_docs)

    print("\n── Stage 2: After Cross-Encoder Rerank ──")
    for doc, score in ranked:
        print(f"  [{score:.3f}]  {doc}")