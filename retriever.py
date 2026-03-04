from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def cosine_score(query_emb, doc_embs):
    """
    Cosine similarity between query and all documents.
    Returns scores in range [-1, 1]. Higher = more similar.

    Best for: General purpose RAG. Works well for most setups.
    """
    scores = cosine_similarity([query_emb], doc_embs)[0]
    return scores


def dot_product_score(query_emb, doc_embs):
    """
    Dot product after L2 normalization.
    Equivalent to cosine similarity but faster for large collections.

    Best for: Large-scale vector databases like FAISS where speed matters.
    """
    query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-9)
    docs_norm  = doc_embs  / (np.linalg.norm(doc_embs, axis=1, keepdims=True) + 1e-9)
    scores     = np.dot(docs_norm, query_norm)
    return scores


def get_top_k(scores, documents, k=3):
    """
    Return top-k (document, score) pairs sorted by score descending.
    """
    top_idx = np.argsort(scores)[::-1][:k]
    return [(documents[i], float(scores[i])) for i in top_idx]


def print_results(results, method_name):
    print(f"\n── {method_name} ──")
    for doc, score in results:
        print(f"  [{score:.3f}]  {doc}")


if __name__ == "__main__":
    from embedder import load_model, embed_documents, embed_query, load_documents

    docs      = load_documents()
    model     = load_model("minilm")
    doc_embs  = embed_documents(model, docs)
    query     = "how does login work?"
    query_emb = embed_query(model, query)

    print(f'\nQuery: "{query}"\n')

    # Cosine
    cos_scores = cosine_score(query_emb, doc_embs)
    print_results(get_top_k(cos_scores, docs), "Cosine Similarity")

    # Dot product
    dot_scores = dot_product_score(query_emb, doc_embs)
    print_results(get_top_k(dot_scores, docs), "Dot Product")