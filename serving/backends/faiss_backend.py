"""Backend FAISS -- in-process, tanpa server terpisah. Default untuk dev/demo
dan knowledge base kecil (puluhan-ratusan dokumen seperti banking_faq.json).

Install: pip install faiss-cpu
"""

import numpy as np

from .base import VectorBackend


class FAISSBackend(VectorBackend):
    def __init__(self):
        self.index = None

    def build(self, embeddings: np.ndarray) -> None:
        try:
            import faiss
        except ImportError:
            raise ImportError("Install faiss-cpu: pip install faiss-cpu")

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # Inner Product = cosine sim (normalized vectors)
        self.index.add(embeddings)

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        scores, indices = self.index.search(query_embedding, top_k)
        return [
            (int(idx), float(score))
            for score, idx in zip(scores[0], indices[0])
            if idx >= 0
        ]
