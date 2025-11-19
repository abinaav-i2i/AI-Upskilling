# src/qdrant_utils.py
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams
from typing import List, Dict

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "listings"

def get_client(url: str = QDRANT_URL) -> QdrantClient:
    """
    Return a QdrantClient pointed at the given URL.
    """
    return QdrantClient(url=url)

def create_collection(client: QdrantClient, vector_size: int, distance: str = "Cosine"):
    """
    Create or recreate the collection using the modern qdrant-client API.
    Uses 'vectors_config' keyword which newer clients expect.
    WARNING: recreate_collection will drop existing data in that collection.
    """
    vectors_conf = VectorParams(size=vector_size, distance=distance)
    try:
        # try to recreate to ensure a fresh collection for development
        client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=vectors_conf
        )
    except TypeError:
        # Some older clients may still accept create_collection only
        try:
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=vectors_conf
            )
        except Exception as e:
            raise
    except Exception:
        # fallback to create if recreate isn't available
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=vectors_conf
        )

def upsert_points(client: QdrantClient, ids: List[str], vectors, payloads: List[Dict], batch_size: int = 256):
    """
    Upsert points in batches. Points must be list of dicts with keys 'id','vector','payload'
    """
    n = len(ids)
    for i in range(0, n, batch_size):
        chunk_ids = ids[i : i + batch_size]
        chunk_vecs = vectors[i : i + batch_size]
        chunk_payloads = payloads[i : i + batch_size]
        points = [
            {"id": chunk_ids[j], "vector": chunk_vecs[j].tolist(), "payload": chunk_payloads[j]}
            for j in range(len(chunk_ids))
        ]
        client.upsert(collection_name=COLLECTION_NAME, points=points)
