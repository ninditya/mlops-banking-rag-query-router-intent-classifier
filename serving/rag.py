"""Backward-compat shim -- implementasi RAG pindah ke rag_engine.py, yang
sekarang bisa switch vector backend (FAISS/Qdrant) lewat env var RAG_BACKEND.
Import lama (`from rag import BankingRAG`) tetap jalan tanpa perlu ubah
inference.py atau evaluation/generate_dataset.py.
"""

from rag_engine import RAGEngine as BankingRAG

__all__ = ["BankingRAG"]
