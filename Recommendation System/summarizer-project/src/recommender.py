# src/recommender.py
"""
Retrieval + optional reranker utilities.

Provides:
- recommend_from_text: fast ANN retrieval using precomputed sentence-transformer embeddings + Qdrant.
- rerank_hits: safe Cross-Encoder reranking that returns a list of dicts
               {"hit": ScoredPoint, "rerank_score": float} sorted by rerank_score desc.
"""
from qdrant_client import QdrantClient
from src.embedder import Embedder
from src.qdrant_utils import get_client
from typing import List, Dict, Optional
import numpy as np

# Default models and collection name
DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
COLLECTION = "listings"

# Instantiate embedder (bi-encoder) once
embedder = Embedder()

def recommend_from_text(query_text: str, top_k: int = 5, client: QdrantClient = None, filter_payload: Dict = None):
    """
    Do a vector search in Qdrant and return the top_k hits (qdrant-client ScoredPoint objects).
    Args:
      - query_text: user query string
      - top_k: number of results to return from Qdrant (ANN search)
      - client: optional QdrantClient instance (created if not passed)
      - filter_payload: optional Qdrant filter dict
    Returns:
      - list of ScoredPoint objects (as returned by qdrant_client.search)
    """
    client = client or get_client()
    qvec = embedder.embed_texts([query_text])[0].tolist()
    search_kwargs = dict(
        collection_name=COLLECTION,
        query_vector=qvec,
        limit=top_k,
        with_payload=True,
    )
    if filter_payload:
        search_kwargs['query_filter'] = filter_payload

    hits = client.search(**search_kwargs)
    return hits

def rerank_hits(query: str, hits: List, rerank_model: str = DEFAULT_RERANK_MODEL, top_k: Optional[int] = None, device: str = "cpu"):
    """
    Rerank `hits` (list of qdrant ScoredPoint) with a Cross-Encoder.
    IMPORTANT: This function does NOT mutate ScoredPoint objects (Pydantic models).
    Instead it returns a list of dicts: [{"hit": ScoredPoint, "rerank_score": float}, ...]
    sorted by rerank_score descending.

    Args:
      - query: the original user query string
      - hits: list of ScoredPoint objects from qdrant_client.search()
      - rerank_model: cross-encoder model name
      - top_k: if provided, return only top_k after reranking
      - device: "cpu" or "cuda"
    Returns:
      - list of dicts with keys "hit" and "rerank_score"
    """
    # Lazy import to avoid heavy import cost when rerank not used
    from sentence_transformers import CrossEncoder

    if not hits:
        return []

    # Build candidate texts (prefer payload['text'] where present)
    candidate_texts = []
    for h in hits:
        payload = getattr(h, "payload", {}) or {}
        ctext = payload.get("text") or payload.get("description") or payload.get("title") or str(payload)
        # trim very long texts to avoid tokenizer limits
        candidate_texts.append(str(ctext)[:4000])

    # CrossEncoder expects device int for CUDA or "cpu"
    device_arg = 0 if device == "cuda" else "cpu"
    cross = CrossEncoder(rerank_model, device=device_arg)

    # Build pairs and predict scores
    pairs = [(query, text) for text in candidate_texts]
    scores = cross.predict(pairs, show_progress_bar=False)

    # Build non-mutating scored list
    scored = [{"hit": h, "rerank_score": float(s)} for h, s in zip(hits, scores)]
    scored_sorted = sorted(scored, key=lambda x: x["rerank_score"], reverse=True)

    if top_k:
        return scored_sorted[:top_k]
    return scored_sorted
