# src/embedder.py
from sentence_transformers import SentenceTransformer
import numpy as np

MODEL_NAME = "all-MiniLM-L6-v2"  # fast, small, good quality

class Embedder:
    def __init__(self, model_name: str = MODEL_NAME):
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts, batch_size: int = 32):
        """
        texts: list[str]
        returns: numpy.ndarray shape (n, dim)
        """
        vectors = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True, batch_size=batch_size)
        return vectors
