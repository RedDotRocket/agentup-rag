"""
Abstract base classes for embedding and vector storage backends.

This module defines the interfaces that all backend implementations must follow,
ensuring consistent behavior across different providers.
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Optional

from ..models import SearchResult, VectorDocument


class EmbeddingBackend(ABC):
    """Abstract base class for embedding backends.

    Embedding backends are responsible for converting text into vector embeddings
    that can be used for semantic search and similarity comparisons.
    """

    def __init__(self, config: dict[str, Any]):
        """Initialize the embedding backend with configuration.

        Args:
            config: Backend-specific configuration dictionary
        """
        self.config = config
        self._validate_config()

    @abstractmethod
    def _validate_config(self) -> None:
        """Validate the configuration for this backend.

        Raises:
            ValueError: If configuration is invalid
        """
        pass

    @abstractmethod
    async def embed_text(self, text: str) -> list[float]:
        """Generate embedding for a single text.

        Args:
            text: Input text to embed

        Returns:
            List of floats representing the embedding vector

        Raises:
            EmbeddingError: If embedding generation fails
        """
        pass

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts efficiently.

        Args:
            texts: List of input texts to embed

        Returns:
            List of embedding vectors, one for each input text

        Raises:
            EmbeddingError: If embedding generation fails
        """
        pass

    async def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this backend.

        Returns:
            Integer dimension of embedding vectors
        """
        # Default implementation: embed a test string and check dimension
        test_embedding = await self.embed_text("test")
        return len(test_embedding)

    async def close(self) -> None:
        """Clean up resources used by the backend.

        This method should be called when the backend is no longer needed.
        """
        pass


class VectorStoreBackend(ABC):
    """Abstract base class for vector storage backends.

    Vector store backends are responsible for storing and searching vector embeddings
    with associated metadata and content.
    """

    def __init__(self, config: dict[str, Any]):
        """Initialize the vector store backend with configuration.

        Args:
            config: Backend-specific configuration dictionary
        """
        self.config = config
        self._validate_config()

    @abstractmethod
    def _validate_config(self) -> None:
        """Validate the configuration for this backend.

        Raises:
            ValueError: If configuration is invalid
        """
        pass

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the vector store backend.

        This method should set up any necessary connections, indices, or
        data structures required for the backend to function.

        Raises:
            VectorStoreError: If initialization fails
        """
        pass

    @abstractmethod
    async def add_vectors(
        self,
        vectors: list[VectorDocument],
        collection: Optional[str] = None
    ) -> None:
        """Add vector documents to the store.

        Args:
            vectors: List of vector documents to add
            collection: Optional collection name to add vectors to

        Raises:
            VectorStoreError: If adding vectors fails
        """
        pass

    @abstractmethod
    async def search(
        self,
        query_vector: list[float],
        k: int = 5,
        collection: Optional[str] = None,
        filters: Optional[dict[str, Any]] = None
    ) -> SearchResult:
        """Search for similar vectors.

        Args:
            query_vector: Vector to search for
            k: Number of results to return
            collection: Optional collection to search in
            filters: Optional metadata filters to apply

        Returns:
            SearchResult containing matching documents and scores

        Raises:
            VectorStoreError: If search fails
        """
        pass

    @abstractmethod
    async def delete_vectors(
        self,
        vector_ids: list[str],
        collection: Optional[str] = None
    ) -> None:
        """Delete vectors by their IDs.

        Args:
            vector_ids: List of vector IDs to delete
            collection: Optional collection to delete from

        Raises:
            VectorStoreError: If deletion fails
        """
        pass

    @abstractmethod
    async def get_collection_stats(self, collection: Optional[str] = None) -> dict[str, Any]:
        """Get statistics about a collection.

        Args:
            collection: Collection name (None for default)

        Returns:
            Dictionary containing collection statistics

        Raises:
            VectorStoreError: If getting stats fails
        """
        pass

    @abstractmethod
    async def list_collections(self) -> list[str]:
        """List all available collections.

        Returns:
            List of collection names

        Raises:
            VectorStoreError: If listing collections fails
        """
        pass

    @abstractmethod
    async def create_collection(
        self,
        name: str,
        dimension: int,
        description: Optional[str] = None
    ) -> None:
        """Create a new collection.

        Args:
            name: Collection name
            dimension: Vector dimension for this collection
            description: Optional collection description

        Raises:
            VectorStoreError: If collection creation fails
        """
        pass

    @abstractmethod
    async def delete_collection(self, name: str) -> None:
        """Delete a collection and all its vectors.

        Args:
            name: Collection name to delete

        Raises:
            VectorStoreError: If collection deletion fails
        """
        pass

    async def close(self) -> None:
        """Clean up resources used by the backend.

        This method should be called when the backend is no longer needed.
        """
        pass


class BackendError(Exception):
    """Base exception for backend errors."""
    pass


class EmbeddingError(BackendError):
    """Exception raised for embedding backend errors."""
    pass


class VectorStoreError(BackendError):
    """Exception raised for vector store backend errors."""
    pass
