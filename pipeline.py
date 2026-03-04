"""
Full RAG Pipeline
─────────────────
Stage 1 : Embed query + documents
Stage 2 : Hybrid retrieval (BM25 + cosine) → top-k candidates
Stage 3 : Cross-encoder reranking → best matches
Stage 4 : Build prompt → send to LLM
"""

import numpy as np
from embedder       import load_model, embed_documents, embed_query, load_documents
from bm25_retriever import build_bm25
from hybrid         import hybrid_score
from reranker       import load_reranker, rerank


def build_prompt(query, context_docs):
    """
    Build the final prompt to send to any LLM.
    context_docs is a list of (document, score) tuples.
    """
    context = "\n".join([f"- {doc}" for doc, _ in context_docs])
    prompt  = f"""Answer based only on the context below. Do not make up information.

Context:
{context}

Question: {query}
Answer:"""
    return prompt


def run_pipeline(query, top_k=3, alpha=0.4, model_name="minilm"):
    """
    Run the full RAG pipeline on a query.

    Args:
        query      : the user's question
        top_k      : how many documents to retrieve
        alpha      : BM25 weight in hybrid scoring (0.4 = 40% BM25, 60% cosine)
        model_name : embedding model to use ("minilm", "bge", or full HF name)

    Returns:
        prompt string ready to send to an LLM
    """
    print("=" * 55)
    print(f'Query: "{query}"')
    print("=" * 55)

    # ── Stage 1: Load and embed ────────────────────────────
    print("\n[1/4] Loading documents and embeddings...")
    docs      = load_documents()
    model     = load_model(model_name)
    doc_embs  = embed_documents(model, docs)
    query_emb = embed_query(model, query)
    bm25      = build_bm25(docs)

    # ── Stage 2: Hybrid retrieval ──────────────────────────
    print(f"\n[2/4] Hybrid retrieval  (alpha={alpha})...")
    h_scores  = hybrid_score(query, query_emb, doc_embs, bm25, alpha=alpha)
    top_idx   = np.argsort(h_scores)[::-1][:top_k * 2]   # get 2x, reranker will trim
    candidates = [docs[i] for i in top_idx]

    print(f"  Retrieved {len(candidates)} candidates:")
    for i in top_idx:
        print(f"    [{h_scores[i]:.3f}]  {docs[i][:60]}")

    # ── Stage 3: Rerank ────────────────────────────────────
    print(f"\n[3/4] Reranking with cross-encoder...")
    reranker = load_reranker()
    ranked   = rerank(reranker, query, candidates)[:top_k]

    print(f"  Top {top_k} after reranking:")
    for doc, score in ranked:
        print(f"    [{score:.3f}]  {doc[:60]}")

    # ── Stage 4: Build prompt ──────────────────────────────
    print(f"\n[4/4] Building prompt...")
    prompt = build_prompt(query, ranked)

    print("\n── Final Prompt ──────────────────────────────────")
    print(prompt)
    print("──────────────────────────────────────────────────")
    print("\n✅  Send this prompt to any LLM (OpenAI / Claude / Ollama)")

    return prompt


if __name__ == "__main__":
    # Try different queries
    queries = [
        "how does the login system work?",
        "what database does the system use?",
        "how are anomalies detected in logs?",
    ]

    for q in queries:
        run_pipeline(q, top_k=3, alpha=0.4)
        print("\n")