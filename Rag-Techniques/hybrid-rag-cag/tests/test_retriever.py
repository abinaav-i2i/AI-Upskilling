import os
import sys
import numpy as np


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))


def test_retriever_search_monkeypatch(monkeypatch):
    # Create a small fake Retriever by instantiating and patching internals
    from retriever.query import Retriever

    r = Retriever.__new__(Retriever)
    # fake embedder
    class E:
        def embed_text(self, s):
            return np.array([1.0, 0.0], dtype=np.float32)

    r.embedder = E()
    # fake index with ntotal and search
    class FakeIndex:
        def __init__(self):
            self.ntotal = 2

        def search(self, q, k):
            # return distances and indices
            D = np.array([[0.9, 0.1]], dtype=np.float32)
            I = np.array([[0, 1]], dtype=np.int64)
            return D, I

    r.index = FakeIndex()
    r.meta = [{'id': 'doc-0', 'title': 'A'}, {'id': 'doc-1', 'title': 'B'}]

    hits, emb = r.retrieve('test query', k=2)
    assert isinstance(hits, list)
    assert len(hits) == 2
    assert hits[0]['meta']['id'] == 'doc-0'
    assert emb.shape[0] == 2 or emb.shape[0] == 2
