"""
RAG (Retrieval-Augmented Generation) module untuk Banking Router.
Menggunakan FAISS + SBERT untuk retrieval dari banking_faq.json.
Tidak memerlukan LLM API — jawaban diambil langsung dari knowledge base.

Dependency:
    pip install faiss-cpu
"""

import json
import os
import numpy as np
import logging

logger = logging.getLogger(__name__)


class BankingRAG:
    def __init__(self, faq_path: str, encoder):
        self.encoder = encoder
        self.faq = []
        self.index = None
        self.embeddings = None
        self._load(faq_path)

    def _load(self, faq_path: str):
        try:
            import faiss
        except ImportError:
            raise ImportError("Install faiss-cpu: pip install faiss-cpu")

        with open(faq_path, "r") as f:
            self.faq = json.load(f)

        questions = [item["question"] for item in self.faq]
        logger.info(f"RAG: encoding {len(questions)} FAQ entries ...")
        self.embeddings = self.encoder.encode(
            questions, normalize_embeddings=True, show_progress_bar=False
        ).astype(np.float32)

        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # Inner Product = cosine sim (normalized vectors)
        self.index.add(self.embeddings)
        logger.info(f"RAG: FAISS index built ({dim}-dim, {len(self.faq)} docs)")

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        q_emb = self.encoder.encode(
            [query], normalize_embeddings=True, show_progress_bar=False
        ).astype(np.float32)
        scores, indices = self.index.search(q_emb, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            item = self.faq[idx]
            results.append({
                "id":       item["id"],
                "question": item["question"],
                "answer":   item["answer"],
                "score":    float(score),
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
