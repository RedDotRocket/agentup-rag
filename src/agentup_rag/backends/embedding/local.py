"""
Local embedding backend implementation using sentence-transformers.

This module provides an embedding backend that uses local sentence-transformer models
to generate embeddings. It supports various pre-trained models and can run on
CPU, CUDA, or MPS devices.
"""

import asyncio
import logging
import os
import threading
import time
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from ..base import EmbeddingBackend, EmbeddingError

logger = logging.getLogger(__name__)


class LocalEmbeddingBackend(EmbeddingBackend):
    """Local embedding backend using sentence-transformers.

    This backend uses sentence-transformers to generate embeddings locally.
    It supports various pre-trained models and can run on different devices.

    Popular models:
    - all-MiniLM-L6-v2 (384 dimensions, fast, good quality)
    - all-mpnet-base-v2 (768 dimensions, high quality)
    - multi-qa-MiniLM-L6-cos-v1 (384 dimensions, optimized for semantic search)
    - paraphrase-MiniLM-L6-v2 (384 dimensions, good for paraphrase detection)
    """

    def __init__(self, config: dict[str, Any]):
        """Initialize the local embedding backend.

        Args:
            config: Configuration dictionary with local model settings
        """
        super().__init__(config)

        self.model_name = config["model"]
        self.device = config.get("device", "cpu")
        self.batch_size = config.get("batch_size", 32)
        self.cache_dir = config.get("cache_dir")
        self.normalize_embeddings = config.get("normalize_embeddings", True)

        # Model instance
        self._model: SentenceTransformer = None
        self._model_lock = threading.Lock()

        self._model_loaded = False

        # Model dimensions mapping (common models)
        self._known_dimensions = {
            "all-MiniLM-L6-v2": 384,
            "all-mpnet-base-v2": 768,
            "multi-qa-MiniLM-L6-cos-v1": 384,
            "paraphrase-MiniLM-L6-v2": 384,
            "all-MiniLM-L12-v2": 384,
            "all-distilroberta-v1": 768,
            "paraphrase-distilroberta-base-v1": 768,
            "sentence-transformers/all-MiniLM-L6-v2": 384,
            "sentence-transformers/all-mpnet-base-v2": 768,
        }

        logger.info(f"Initialized local embedding backend with model: {self.model_name}")

    def _validate_config(self) -> None:
        """Validate the local embedding configuration.

        Raises:
            ValueError: If configuration is invalid
        """
        if not self.config.get("model"):
            raise ValueError("Model name is required for local embedding backend")

        device = self.config.get("device", "cpu")
        valid_devices = ["cpu", "cuda", "mps"]
        if device not in valid_devices:
            raise ValueError(f"Device must be one of: {valid_devices}")

        batch_size = self.config.get("batch_size", 32)
        if batch_size <= 0 or batch_size > 1000:
            raise ValueError("Batch size must be between 1 and 1000")

    def _load_model(self) -> None:
        """Load the sentence transformer model.

        This method is thread-safe and will only load the model once.

        Raises:
            EmbeddingError: If model loading fails
        """
        if self._model_loaded:
            return

        with self._model_lock:

            if self._model_loaded:
                return

            try:
                start_time = time.time()
                logger.info(f"Loading sentence transformer model: {self.model_name}")

                # Set up cache directory if specified
                cache_folder = None
                if self.cache_dir:
                    cache_folder = os.path.expanduser(self.cache_dir)
                    os.makedirs(cache_folder, exist_ok=True)

                # Load the model
                self._model = SentenceTransformer(
                    self.model_name,
                    device=self.device,
                    cache_folder=cache_folder
                )

                # Verify the model works by encoding a test sentence
                test_embedding = self._model.encode(["test"], convert_to_numpy=True)
                if test_embedding is None or len(test_embedding) == 0:
                    raise EmbeddingError("Model failed to produce test embedding")

                self._model_loaded = True
                load_time = (time.time() - start_time) * 1000

                logger.info(f"Loaded model {self.model_name} on {self.device} in {load_time:.2f}ms")
                logger.info(f"Model embedding dimension: {self.get_embedding_dimension_sync()}")

            except Exception as e:
                error_msg = f"Failed to load sentence transformer model '{self.model_name}': {e}"
                logger.error(error_msg)
                raise EmbeddingError(error_msg) from e

    async def embed_text(self, text: str) -> list[float]:
        """Generate embedding for a single text.

        Args:
            text: Input text to embed

        Returns:
            Embedding vector as list of floats

        Raises:
            EmbeddingError: If embedding generation fails
        """
        if not text or not text.strip():
            raise EmbeddingError("Cannot embed empty text")

        # Use batch method for consistency
        embeddings = await self.embed_batch([text.strip()])
        return embeddings[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts efficiently.

        This method processes texts in batches and runs the computation
        in a thread pool to avoid blocking the event loop.

        Args:
            texts: List of input texts to embed

        Returns:
            List of embedding vectors, one for each input text

        Raises:
            EmbeddingError: If embedding generation fails
        """
        if not texts:
            return []

        # Filter out empty texts and track original indices
        filtered_texts = []
        text_indices = []

        for i, text in enumerate(texts):
            if text and text.strip():
                filtered_texts.append(text.strip())
                text_indices.append(i)

        if not filtered_texts:
            raise EmbeddingError("No valid texts to embed")

        try:
            # Load model if not already loaded
            if not self._model_loaded:
                # Run model loading in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self._load_model)

            # Generate embeddings in thread pool
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None,
                self._generate_embeddings_sync,
                filtered_texts
            )

            # Create result array with correct size, filling empty texts with None
            result = [None] * len(texts)
            for i, embedding in enumerate(embeddings):
                original_index = text_indices[i]
                result[original_index] = embedding

            # Replace None with zero embeddings for consistency
            dimension = self.get_embedding_dimension_sync()
            for i, embedding in enumerate(result):
                if embedding is None:
                    result[i] = [0.0] * dimension
                    logger.warning(f"Text at index {i} was empty, using zero embedding")

            return result

        except Exception as e:
            error_msg = f"Failed to generate embeddings: {e}"
            logger.error(error_msg)
            raise EmbeddingError(error_msg) from e

    def _generate_embeddings_sync(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings synchronously (for use in thread pool).

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        if not self._model:
            raise EmbeddingError("Model not loaded")

        start_time = time.time()

        # Process in batches to manage memory
        all_embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]

            # Generate embeddings for this batch
            batch_embeddings = self._model.encode(
                batch_texts,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize_embeddings,
                show_progress_bar=False  # Avoid progress bars in production
            )

            # Convert to list of lists
            for embedding in batch_embeddings:
                all_embeddings.append(embedding.tolist())

        processing_time = (time.time() - start_time) * 1000
        logger.debug(f"Generated {len(all_embeddings)} embeddings in {processing_time:.2f}ms")

        return all_embeddings

    async def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this backend.

        Returns:
            Integer dimension of embedding vectors
        """
        # Try to get from known dimensions first
        if self.model_name in self._known_dimensions:
            return self._known_dimensions[self.model_name]

        # If model is loaded, get actual dimension
        if self._model_loaded and self._model:
            return self._model.get_sentence_embedding_dimension()

        # Load model to get dimension
        if not self._model_loaded:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._load_model)

        return self._model.get_sentence_embedding_dimension()

    def get_embedding_dimension_sync(self) -> int:
        """Synchronous version of get_embedding_dimension for internal use."""
        # Try to get from known dimensions first
        if self.model_name in self._known_dimensions:
            return self._known_dimensions[self.model_name]

        # If model is loaded, get actual dimension
        if self._model_loaded and self._model:
            return self._model.get_sentence_embedding_dimension()

        # Load model to get dimension
        if not self._model_loaded:
            self._load_model()

        return self._model.get_sentence_embedding_dimension()

    async def close(self) -> None:
        """Clean up resources used by the backend."""
        with self._model_lock:

            if self._model:
                # Move model to CPU and clear CUDA cache if using CUDA
                if self.device.startswith("cuda"):
                    try:
                        import torch
                        self._model = self._model.cpu()
                        torch.cuda.empty_cache()
                        logger.debug("Cleared CUDA cache for local embedding model")
                    except ImportError:
                        pass

                self._model = None
                self._model_loaded = False
                logger.debug("Closed local embedding backend")

    def __del__(self):
        """Cleanup when object is destroyed."""
        if self._model_loaded:
            logger.warning("Local embedding backend was not properly closed")

    async def get_usage_stats(self) -> dict[str, Any]:
        """Get usage statistics for monitoring.

        Returns:
            dictionary with usage statistics
        """
        return {
            "backend_type": "local",
            "model": self.model_name,
            "device": self.device,
            "batch_size": self.batch_size,
            "normalize_embeddings": self.normalize_embeddings,
            "model_loaded": self._model_loaded,
            "cache_dir": self.cache_dir,
            "embedding_dimension": self.get_embedding_dimension_sync() if self._model_loaded else "unknown",
        }

    def get_model_info(self) -> dict[str, Any]:
        """Get detailed information about the loaded model.

        Returns:
            Dictionary with model information
        """
        if not self._model_loaded or not self._model:
            return {"error": "Model not loaded"}

        try:
            return {
                "model_name": self.model_name,
                "embedding_dimension": self._model.get_sentence_embedding_dimension(),
                "max_seq_length": getattr(self._model, 'max_seq_length', 'unknown'),
                "device": str(self._model.device),
                "normalize_embeddings": self.normalize_embeddings,
            }
        except Exception as e:
            return {"error": f"Failed to get model info: {e}"}

    def is_available(self) -> bool:
        """Check if sentence-transformers is available.

        Returns:
            True if sentence-transformers can be imported
        """
        try:
            import sentence_transformers
            return True
        except ImportError:
            return False
