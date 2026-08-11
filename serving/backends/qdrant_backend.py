"""Backend Qdrant -- vector DB terpisah dari proses app. Cocok untuk
production: knowledge base besar, butuh filtering/payload, atau index yang
di-share antar instance app (FAISS in-process tidak bisa di-share).

Default jalan in-memory (embedded, tanpa server) kalau QDRANT_URL tidak
di-set -- supaya tetap bisa dites lokal tanpa infra tambahan. Set QDRANT_URL
(+ opsional QDRANT_API_KEY) untuk pakai server sungguhan (Qdrant Cloud atau
self-hosted).

Install: pip install qdrant-client
"""

import os

import numpy as np

from .base import VectorBackend

COLLECTION_NAME = "banking_faq"


class QdrantBackend(VectorBackend):
    def __init__(self):
        self.client = None

    def _connect(self):
        try:
            from qdrant_client import QdrantClient
        except ImportError:
            raise ImportError("Install qdrant-client: pip install qdrant-client")

        url = os.getenv("QDRANT_URL")
        if url:
            return QdrantClient(url=url, api_key=os.getenv("QDRANT_API_KEY"))
        return QdrantClient(location=":memory:")

    def build(self, embeddings: np.ndarray) -> None:
        from qdrant_client.models import Distance, PointStruct, VectorParams

        self.client = self._connect()
        dim = embeddings.shape[1]

        if self.client.collection_exists(COLLECTION_NAME):
            self.client.delete_collection(COLLECTION_NAME)
        self.client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        points = [
            PointStruct(id=i, vector=embeddings[i].tolist(), payload={"doc_index": i})
            for i in range(len(embeddings))
        ]
        self.client.upsert(collection_name=COLLECTION_NAME, points=points)

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        result = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_embedding[0].tolist(),
            limit=top_k,
        )
        return [(point.payload["doc_index"], float(point.score)) for point in result.points]
