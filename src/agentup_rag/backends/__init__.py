"""
Backend implementations for embedding and vector storage.

This module provides abstract base classes and concrete implementations for:
- Embedding backends (OpenAI, local models)
- Vector store backends (memory, Chroma, Pinecone, Weaviate)
"""

from .base import EmbeddingBackend, VectorStoreBackend
from .factory import BackendFactory

__all__ = [
    "EmbeddingBackend",
    "VectorStoreBackend", 
    "BackendFactory",
]