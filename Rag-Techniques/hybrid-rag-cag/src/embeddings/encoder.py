from sentence_transformers import SentenceTransformer
import numpy as np
from ..core.config import settings


class Embedder:
    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.EMBED_MODEL
        self.model = SentenceTransformer(self.model_name)

    def embed_texts(self, texts: list) -> np.ndarray:
        arr = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        # normalize to unit vectors if not already
        norms = (arr**2).sum(axis=1, keepdims=True)**0.5
        norms[norms==0] = 1.0
        return (arr / norms).astype('float32')

    def embed_text(self, text: str):
        return self.embed_texts([text])[0]