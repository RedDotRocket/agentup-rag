"""Embedding backend implementations."""

__all__ = []

try:
    from .openai import OpenAIEmbeddingBackend
    __all__.append("OpenAIEmbeddingBackend")
except ImportError:
    pass

try:
    from .local import LocalEmbeddingBackend  
    __all__.append("LocalEmbeddingBackend")
except ImportError:
    pass