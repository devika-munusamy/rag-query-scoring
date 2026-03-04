from sentence_transformers import SentenceTransformer
import numpy as np

# Supported model names
MODELS = {
    "minilm":   "all-MiniLM-L6-v2",
    "bge":      "BAAI/bge-large-en-v1.5",
    "bge-m3":   "BAAI/bge-m3",
}


def load_model(name="minilm"):
    """
    Load an embedding model by short name.

    Available:
        "minilm"  -> all-MiniLM-L6-v2       (fast, lightweight)
        "bge"     -> BAAI/bge-large-en-v1.5  (best open source)
        "bge-m3"  -> BAAI/bge-m3             (multilingual)

    Or pass a full HuggingFace model name directly.
    """
    model_name = MODELS.get(name, name)
    print(f"Loading model: {model_name}")
    return SentenceTransformer(model_name)


def embed_documents(model, documents):
    """
    Embed a list of document strings.
    Returns numpy array of shape (n_docs, embedding_dim).
    """
    return model.encode(documents, show_progress_bar=False)


def embed_query(model, query, use_bge_prefix=False):
    """
    Embed a single query string.
    Set use_bge_prefix=True when using BGE models — they require a special prefix.
    """
    if use_bge_prefix:
        query = f"Represent this sentence for searching relevant passages: {query}"
    return model.encode(query)


def load_documents(file_path="documents/sample_docs.txt"):
    """
    Load documents from a text file — one document per line.
    """
    with open(file_path, "r") as f:
        docs = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(docs)} documents from {file_path}")
    return docs


if __name__ == "__main__":
    docs  = load_documents()
    model = load_model("minilm")

    doc_embs   = embed_documents(model, docs)
    query_emb  = embed_query(model, "how does login work?")

    print(f"\nDocument embeddings shape : {doc_embs.shape}")
    print(f"Query embedding shape     : {query_emb.shape}")
    print(f"Embedding dimensions      : {doc_embs.shape[1]}")