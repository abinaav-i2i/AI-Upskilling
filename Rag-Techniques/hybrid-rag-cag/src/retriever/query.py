import json
import numpy as np
import faiss
from ..embeddings.encoder import Embedder
from ..core.config import settings
from ..core.logger import get_logger
from typing import Tuple, List


logger = get_logger('retriever')


class Retriever:
    def __init__(self, index_path: str = None, meta_path: str = None):
        self.index_path = index_path or settings.FAISS_INDEX_PATH
        self.meta_path = meta_path or settings.FAISS_META_PATH
        self.embedder = Embedder()
        self.index = None
        self.meta = []
        self._load()

    def _load(self):
        try:
            logger.info('Loading FAISS index from %s', self.index_path)
            self.index = faiss.read_index(self.index_path)
        except Exception:
            logger.exception('Could not read FAISS index')
            raise
        try:
            self.meta = []
            with open(self.meta_path, 'r', encoding='utf-8') as f:
                for line in f:
                    self.meta.append(json.loads(line))
            logger.info('Loaded %d meta entries', len(self.meta))
        except Exception:
            logger.exception('Failed to load meta file')
            raise

    def retrieve(self, query: str, k: int = 5) -> Tuple[List[dict], np.ndarray]:
        """Return top-k hits and the query embedding.

        Raises when index is not loaded. If k is larger than index size, it will be clamped.
        """
        if self.index is None:
            raise RuntimeError('FAISS index not loaded')

        q_emb = self.embedder.embed_text(query).astype('float32')
        n_total = int(self.index.ntotal)
        k = max(1, min(k, n_total if n_total > 0 else 1))
        try:
            D, I = self.index.search(np.array([q_emb]), k)
        except Exception:
            logger.exception('FAISS search failed')
            raise

        hits = []
        for score, idx in zip(D[0], I[0]):
            if int(idx) < 0:
                continue
            try:
                m = self.meta[int(idx)]
            except Exception:
                logger.debug('Missing meta for idx %s', idx)
                m = {}
            hits.append({'meta': m, 'score': float(score)})
        return hits, q_emb
