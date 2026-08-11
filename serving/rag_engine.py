"""
RAG engine untuk Banking Router. Encoding pakai SBERT, penyimpanan/pencarian
vektor didelegasikan ke backend yang bisa di-switch (lihat backends/):

    RAG_BACKEND=faiss    (default) -- in-process, pip install faiss-cpu
    RAG_BACKEND=qdrant              -- vector DB terpisah, pip install qdrant-client
                                        (in-memory kalau QDRANT_URL tidak di-set)

Backend dipilih lewat parameter `backend` atau env var RAG_BACKEND kalau
parameter tidak diberikan.
"""

import json
import os
import logging
import numpy as np

from backends import get_backend

logger = logging.getLogger(__name__)


class RAGEngine:
    def __init__(self, faq_path: str, encoder, backend: str | None = None):
        self.encoder = encoder
        self.faq = []
        self.backend_name = backend or os.getenv("RAG_BACKEND", "faiss")
        self.backend = get_backend(self.backend_name)
        self._load(faq_path)

    def _load(self, faq_path: str):
        with open(faq_path, "r") as f:
            self.faq = json.load(f)

        questions = [item["question"] for item in self.faq]
        logger.info(f"RAG: encoding {len(questions)} FAQ entries ...")
        embeddings = self.encoder.encode(
            questions, normalize_embeddings=True, show_progress_bar=False
        ).astype(np.float32)

        self.backend.build(embeddings)
        logger.info(
            f"RAG: {self.backend_name} index built "
            f"({embeddings.shape[1]}-dim, {len(self.faq)} docs)"
        )

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        q_emb = self.encoder.encode(
            [query], normalize_embeddings=True, show_progress_bar=False
        ).astype(np.float32)
        hits = self.backend.search(q_emb, top_k)
        results = []
        for idx, score in hits:
            item = self.faq[idx]
            results.append({
                "id":       item["id"],
                "question": item["question"],
                "answer":   item["answer"],
                "score":    score,
                "tags":     item.get("tags", []),
            })
        return results

    def answer(self, query: str, top_k: int = 3) -> dict:
        results = self.retrieve(query, top_k=top_k)
        if not results:
            return {
                "answer": "I couldn't find specific information about that. Please contact our support team.",
                "sources": [],
                "top_score": 0.0,
            }
        best = results[0]
        return {
            "answer":    best["answer"],
            "sources":   results,
            "top_score": best["score"],
        }
