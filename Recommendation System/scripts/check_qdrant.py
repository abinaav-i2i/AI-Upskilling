# quick check (run in python)
from qdrant_client import QdrantClient
client = QdrantClient(url="http://localhost:6333")
print("Connected:", client)
# list collections (should succeed)
print("Collections:", client.get_collections().collections)
