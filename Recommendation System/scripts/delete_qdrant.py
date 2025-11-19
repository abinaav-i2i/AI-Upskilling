from qdrant_client import QdrantClient


client = QdrantClient(url="http://localhost:6333")
# remove collection if it exists (safe for development)
try:
    client.delete_collection(collection_name="listings")
    print("Deleted existing 'listings' collection (if it existed).")
except Exception as e:
    print("Delete collection returned:", e)