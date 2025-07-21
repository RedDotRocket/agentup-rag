"""
Backend factory for dynamic instantiation of embedding and vector store backends.

This module provides a factory pattern implementation for creating backend instances
based on configuration. It uses a registry pattern to allow easy addition of new backends.
"""

import logging
from typing import Any, Dict, Type

from ..models import (
    ChromaVectorStoreConfig,
    LocalEmbeddingConfig,
    MemoryVectorStoreConfig,
    OpenAIEmbeddingConfig,
    PineconeVectorStoreConfig,
    WeaviateVectorStoreConfig,
)
from .base import EmbeddingBackend, VectorStoreBackend

logger = logging.getLogger(__name__)


class BackendRegistry:
    """Registry for backend implementations.

    This class maintains a registry of available backend implementations
    and provides methods to register and retrieve them.
    """

    def __init__(self):
        """Initialize empty registries."""
        self._embedding_backends: Dict[str, Type[EmbeddingBackend]] = {}
        self._vector_store_backends: Dict[str, Type[VectorStoreBackend]] = {}

    def register_embedding_backend(self, name: str, backend_class: Type[EmbeddingBackend]) -> None:
        """Register an embedding backend implementation.

        Args:
            name: Backend name (e.g., 'openai', 'local')
            backend_class: Backend implementation class
        """
        if not issubclass(backend_class, EmbeddingBackend):
            raise ValueError(f"Backend class must inherit from EmbeddingBackend")

        self._embedding_backends[name] = backend_class
        logger.debug(f"Registered embedding backend: {name}")

    def register_vector_store_backend(self, name: str, backend_class: Type[VectorStoreBackend]) -> None:
        """Register a vector store backend implementation.

        Args:
            name: Backend name (e.g., 'memory', 'chroma')
            backend_class: Backend implementation class
        """
        if not issubclass(backend_class, VectorStoreBackend):
            raise ValueError(f"Backend class must inherit from VectorStoreBackend")

        self._vector_store_backends[name] = backend_class
        logger.debug(f"Registered vector store backend: {name}")

    def get_embedding_backend(self, name: str) -> Type[EmbeddingBackend]:
        """Get an embedding backend class by name.

        Args:
            name: Backend name

        Returns:
            Backend implementation class

        Raises:
            ValueError: If backend is not registered
        """
        if name not in self._embedding_backends:
            available = list(self._embedding_backends.keys())
            raise ValueError(f"Unknown embedding backend '{name}'. Available: {available}")

        return self._embedding_backends[name]

    def get_vector_store_backend(self, name: str) -> Type[VectorStoreBackend]:
        """Get a vector store backend class by name.

        Args:
            name: Backend name

        Returns:
            Backend implementation class

        Raises:
            ValueError: If backend is not registered
        """
        if name not in self._vector_store_backends:
            available = list(self._vector_store_backends.keys())
            raise ValueError(f"Unknown vector store backend '{name}'. Available: {available}")

        return self._vector_store_backends[name]

    def list_embedding_backends(self) -> list[str]:
        """List all registered embedding backends.

        Returns:
            List of backend names
        """
        return list(self._embedding_backends.keys())

    def list_vector_store_backends(self) -> list[str]:
        """List all registered vector store backends.

        Returns:
            List of backend names
        """
        return list(self._vector_store_backends.keys())


# Global registry instance
_registry = BackendRegistry()


class BackendFactory:
    """Factory for creating backend instances.

    This factory uses the global registry to create backend instances
    based on configuration and backend type.
    """

    @staticmethod
    def create_embedding_backend(backend_type: str, config: Dict[str, Any]) -> EmbeddingBackend:
        """Create an embedding backend instance.

        Args:
            backend_type: Type of backend to create (e.g., 'openai', 'local')
            config: Configuration dictionary for the backend

        Returns:
            Configured embedding backend instance

        Raises:
            ValueError: If backend type is unknown or configuration is invalid
        """
        logger.info(f"Creating embedding backend: {backend_type}")

        # Get backend class from registry
        backend_class = _registry.get_embedding_backend(backend_type)

        # Validate and parse configuration
        validated_config = BackendFactory._validate_embedding_config(backend_type, config)

        # Create and return backend instance
        try:
            backend = backend_class(validated_config.model_dump())
            logger.info(f"Successfully created embedding backend: {backend_type}")
            return backend
        except Exception as e:
            logger.error(f"Failed to create embedding backend '{backend_type}': {e}")
            raise ValueError(f"Failed to create embedding backend '{backend_type}': {e}") from e

    @staticmethod
    def create_vector_store_backend(backend_type: str, config: Dict[str, Any]) -> VectorStoreBackend:
        """Create a vector store backend instance.

        Args:
            backend_type: Type of backend to create (e.g., 'memory', 'chroma')
            config: Configuration dictionary for the backend

        Returns:
            Configured vector store backend instance

        Raises:
            ValueError: If backend type is unknown or configuration is invalid
        """
        logger.info(f"Creating vector store backend: {backend_type}")

        # Get backend class from registry
        backend_class = _registry.get_vector_store_backend(backend_type)

        # Validate and parse configuration
        validated_config = BackendFactory._validate_vector_store_config(backend_type, config)

        # Create and return backend instance
        try:
            backend = backend_class(validated_config.model_dump())
            logger.info(f"Successfully created vector store backend: {backend_type}")
            return backend
        except Exception as e:
            logger.error(f"Failed to create vector store backend '{backend_type}': {e}")
            raise ValueError(f"Failed to create vector store backend '{backend_type}': {e}") from e

    @staticmethod
    def _validate_embedding_config(backend_type: str, config: Dict[str, Any]):
        """Validate configuration for embedding backend.

        Args:
            backend_type: Backend type
            config: Configuration dictionary

        Returns:
            Validated configuration object

        Raises:
            ValueError: If configuration is invalid
        """
        config_classes = {
            "openai": OpenAIEmbeddingConfig,
            "local": LocalEmbeddingConfig,
        }

        if backend_type not in config_classes:
            raise ValueError(f"Unknown embedding backend type: {backend_type}")

        config_class = config_classes[backend_type]

        try:
            return config_class(**config)
        except Exception as e:
            raise ValueError(f"Invalid configuration for {backend_type} embedding backend: {e}") from e

    @staticmethod
    def _validate_vector_store_config(backend_type: str, config: Dict[str, Any]):
        """Validate configuration for vector store backend.

        Args:
            backend_type: Backend type
            config: Configuration dictionary

        Returns:
            Validated configuration object

        Raises:
            ValueError: If configuration is invalid
        """
        config_classes = {
            "memory": MemoryVectorStoreConfig,
            "chroma": ChromaVectorStoreConfig,
            "pinecone": PineconeVectorStoreConfig,
            "weaviate": WeaviateVectorStoreConfig,
        }

        if backend_type not in config_classes:
            raise ValueError(f"Unknown vector store backend type: {backend_type}")

        config_class = config_classes[backend_type]

        try:
            return config_class(**config)
        except Exception as e:
            raise ValueError(f"Invalid configuration for {backend_type} vector store backend: {e}") from e

    @staticmethod
    def list_available_backends() -> Dict[str, list[str]]:
        """List all available backend types.

        Returns:
            Dictionary with 'embedding' and 'vector_store' keys containing lists of backend names
        """
        return {
            "embedding": _registry.list_embedding_backends(),
            "vector_store": _registry.list_vector_store_backends(),
        }


def register_embedding_backend(name: str, backend_class: Type[EmbeddingBackend]) -> None:
    """Register an embedding backend implementation.

    This is a convenience function for registering backends with the global registry.

    Args:
        name: Backend name
        backend_class: Backend implementation class
    """
    _registry.register_embedding_backend(name, backend_class)


def register_vector_store_backend(name: str, backend_class: Type[VectorStoreBackend]) -> None:
    """Register a vector store backend implementation.

    This is a convenience function for registering backends with the global registry.

    Args:
        name: Backend name
        backend_class: Backend implementation class
    """
    _registry.register_vector_store_backend(name, backend_class)


def auto_register_backends() -> None:
    """Automatically register all available backend implementations.

    This function attempts to import and register all known backend implementations.
    It gracefully handles missing dependencies by logging warnings.
    """
    logger.info("Auto-registering available backends...")

    # Try to register embedding backends
    try:
        from .embedding.openai import OpenAIEmbeddingBackend
        register_embedding_backend("openai", OpenAIEmbeddingBackend)
    except ImportError as e:
        logger.warning(f"OpenAI embedding backend not available: {e}")

    try:
        from .embedding.local import LocalEmbeddingBackend
        register_embedding_backend("local", LocalEmbeddingBackend)
    except ImportError as e:
        logger.warning(f"Local embedding backend not available: {e}")

    # Try to register vector store backends
    try:
        from .vector_store.memory import MemoryVectorStoreBackend
        register_vector_store_backend("memory", MemoryVectorStoreBackend)
    except ImportError as e:
        logger.warning(f"Memory vector store backend not available: {e}")

    try:
        from .vector_store.chroma import ChromaVectorStoreBackend
        register_vector_store_backend("chroma", ChromaVectorStoreBackend)
    except ImportError as e:
        logger.warning(f"Chroma vector store backend not available: {e}")

    try:
        from .vector_store.pinecone import PineconeVectorStoreBackend
        register_vector_store_backend("pinecone", PineconeVectorStoreBackend)
    except ImportError as e:
        logger.warning(f"Pinecone vector store backend not available: {e}")

    try:
        from .vector_store.weaviate import WeaviateVectorStoreBackend
        register_vector_store_backend("weaviate", WeaviateVectorStoreBackend)
    except ImportError as e:
        logger.warning(f"Weaviate vector store backend not available: {e}")

    # Log registered backends
    available = BackendFactory.list_available_backends()
    logger.info(f"Registered embedding backends: {available['embedding']}")
    logger.info(f"Registered vector store backends: {available['vector_store']}")


# Auto-register backends when module is imported
auto_register_backends()
