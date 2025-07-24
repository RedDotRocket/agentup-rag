"""Embedding backend implementations."""

__all__ = []

try:
    from .openai import OpenAIEmbeddingBackend  # noqa: F401

    __all__.append("OpenAIEmbeddingBackend")
except ImportError:
    pass

try:
    from .local import LocalEmbeddingBackend  # noqa: F401

    __all__.append("LocalEmbeddingBackend")
except ImportError:
    pass
