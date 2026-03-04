"""
main.py — Run all 5 scoring methods on the same query and compare results.

Usage:
    python main.py
    python main.py --query "what database is used?"
    python main.py --query "how does login work?" --top_k 5
"""

import argparse
import numpy as np
import warnings
import logging

# Suppress noisy model loading warnings (safe to ignore)
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

from embedder       import load_model, embed_documents, embed_query, load_documents
from retriever      import cosine_score, dot_product_score
from bm25_retriever import build_bm25, bm25_score
from hybrid         import hybrid_score
from reranker       import load_reranker, rerank


def print_block(title, results):
    print(f"\n── {title} ──")
    for doc, score in results:
        bar   = "█" * int(score * 20)
        empty = "░" * (20 - int(score * 20))
        print(f"  [{score:.3f}] {bar}{empty}  {doc[:55]}")


def get_top_k(scores, documents, k):
    idx = np.argsort(scores)[::-1][:k]
    return [(documents[i], float(scores[i])) for i in idx]


def run_all(query, top_k=3):
    print("\n" + "=" * 65)
    print(f'  Query : "{query}"')
    print(f'  Top-k : {top_k}')
    print("=" * 65)

    # ── Setup ──────────────────────────────────────────────
    docs      = load_documents()
    model     = load_model("minilm")
    doc_embs  = embed_documents(model, docs)
    query_emb = embed_query(model, query)
    bm25      = build_bm25(docs)

    # ── 1. Cosine Similarity ───────────────────────────────
    cos = cosine_score(query_emb, doc_embs)
    print_block("1. Cosine Similarity", get_top_k(cos, docs, top_k))

    # ── 2. Dot Product ─────────────────────────────────────
    dot = dot_product_score(query_emb, doc_embs)
    print_block("2. Dot Product", get_top_k(dot, docs, top_k))

    # ── 3. BM25 ────────────────────────────────────────────
    bm25s = bm25_score(bm25, query)
    if bm25s.max() == 0:
        print(f"\n── 3. BM25 (Keyword) ──")
        print(f"  [0.000]  No exact keyword match found for this query.")
        print(f"  Tip: BM25 works best with specific terms like 'Redis', 'PostgreSQL', 'login'")
    else:
        print_block("3. BM25 (Keyword)", get_top_k(bm25s, docs, top_k))

    # ── 4. Hybrid ──────────────────────────────────────────
    hyb = hybrid_score(query, query_emb, doc_embs, bm25, alpha=0.4)
    print_block("4. Hybrid (BM25 + Cosine, alpha=0.4)", get_top_k(hyb, docs, top_k))

    # ── 5. Cross-Encoder Reranker ──────────────────────────
    top5_idx  = np.argsort(cos)[::-1][:5]
    top5_docs = [docs[i] for i in top5_idx]
    reranker  = load_reranker()
    ranked    = rerank(reranker, query, top5_docs)[:top_k]
    print_block("5. Cross-Encoder Reranker", ranked)

    # ── Summary ────────────────────────────────────────────
    print("\n── Top-1 Summary ──")
    print(f"  Cosine       [{cos.max():.3f}]  {docs[cos.argmax()][:55]}")
    print(f"  Dot Product  [{dot.max():.3f}]  {docs[dot.argmax()][:55]}")
    if bm25s.max() > 0:
        print(f"  BM25         [{bm25s.max():.3f}]  {docs[bm25s.argmax()][:55]}")
    else:
        print(f"  BM25         [0.000]  no keyword match")
    print(f"  Hybrid       [{hyb.max():.3f}]  {docs[hyb.argmax()][:55]}")
    print(f"  CrossEncoder [{ranked[0][1]:.3f}]  {ranked[0][0][:55]}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Query Scoring Comparison")
    parser.add_argument("--query",  type=str, default="how does login work?")
    parser.add_argument("--top_k",  type=int, default=3)
    args = parser.parse_args()

    run_all(args.query, args.top_k)