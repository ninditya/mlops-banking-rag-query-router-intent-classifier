"""Interface vector backend untuk RAGEngine.

Kontrak: embedding yang masuk sudah dinormalisasi (SBERT normalize_embeddings=True),
jadi cosine similarity == inner product. Semua backend harus mengembalikan
skor dalam konvensi yang sama: lebih tinggi = lebih mirip.
"""

from abc import ABC, abstractmethod

import numpy as np


class VectorBackend(ABC):
    @abstractmethod
    def build(self, embeddings: np.ndarray) -> None:
        """Index seluruh embedding sekaligus. Baris ke-i merepresentasikan
        dokumen ke-i di banking_faq.json (urutan harus dipertahankan --
        RAGEngine memetakan index hasil search kembali ke self.faq[index])."""

    @abstractmethod
    def search(self, query_embedding: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        """Cari top_k dokumen paling mirip. Return list (doc_index, score)
        terurut dari yang paling mirip, panjang <= top_k."""
