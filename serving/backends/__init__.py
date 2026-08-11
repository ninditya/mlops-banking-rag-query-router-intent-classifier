from .base import VectorBackend
from .faiss_backend import FAISSBackend
from .qdrant_backend import QdrantBackend

BACKENDS: dict[str, type[VectorBackend]] = {
    "faiss": FAISSBackend,
    "qdrant": QdrantBackend,
}


def get_backend(name: str) -> VectorBackend:
    try:
        return BACKENDS[name]()
    except KeyError:
        raise ValueError(f"RAG_BACKEND tidak dikenal: '{name}'. Pilihan: {list(BACKENDS)}")
