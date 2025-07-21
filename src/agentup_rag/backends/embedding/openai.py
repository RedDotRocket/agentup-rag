"""
OpenAI embedding backend implementation.

This module provides an embedding backend that uses OpenAI's embedding models
via their API. It includes features like batch processing, rate limiting,
retry logic, and efficient HTTP connection management.
"""

import asyncio
import logging
import time
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from ..base import EmbeddingBackend, EmbeddingError

logger = logging.getLogger(__name__)


class OpenAIEmbeddingBackend(EmbeddingBackend):
    """OpenAI embedding backend with async HTTP client and batch processing.

    This backend uses OpenAI's embedding API to generate vector embeddings
    for text. It includes optimizations for batch processing, rate limiting,
    and error handling.

    Supported models:
    - text-embedding-3-small (1536 dimensions, faster, cost-effective)
    - text-embedding-3-large (3072 dimensions, higher quality)
    - text-embedding-ada-002 (legacy support)
    """

    def __init__(self, config: dict[str, Any]):
        """Initialize the OpenAI embedding backend.

        Args:
            config: Configuration dictionary with OpenAI settings
        """
        # Model dimensions mapping - must be set before super().__init__
        self._model_dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        
        # Initialize client to None to avoid AttributeError in __del__
        self._client: httpx.AsyncClient = None
        
        super().__init__(config)

        self.api_key = config["api_key"]
        self.model = config["model"]
        self.organization = config.get("organization")
        self.batch_size = config.get("batch_size", 100)
        self.rate_limit = config.get("rate_limit", 60)  # requests per minute
        self.max_retries = config.get("max_retries", 3)
        self.backoff_factor = config.get("backoff_factor", 2.0)
        self.timeout = config.get("timeout", 30)

        # Rate limiting state
        self._request_times: list[float] = []
        self._rate_limit_lock = asyncio.Lock()

        # HTTP client setup
        self._setup_client()

        logger.info(f"Initialized OpenAI embedding backend with model: {self.model}")

    def _validate_config(self) -> None:
        """Validate the OpenAI configuration.

        Raises:
            ValueError: If configuration is invalid
        """
        if not self.config.get("api_key"):
            raise ValueError("OpenAI API key is required")

        if not self.config.get("model"):
            raise ValueError("OpenAI model is required")

        model = self.config["model"]
        if model not in self._model_dimensions:
            supported = list(self._model_dimensions.keys())
            raise ValueError(f"Unsupported model '{model}'. Supported models: {supported}")

        batch_size = self.config.get("batch_size", 100)
        if batch_size <= 0 or batch_size > 2048:
            raise ValueError("Batch size must be between 1 and 2048")

    def _setup_client(self) -> None:
        """Set up the async HTTP client with proper headers and settings."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "agentup-rag/0.1.0",
        }

        if self.organization:
            headers["OpenAI-Organization"] = self.organization

        # Configure timeouts and connection pooling
        timeout = httpx.Timeout(
            connect=10.0,
            read=self.timeout,
            write=10.0,
            pool=30.0
        )

        limits = httpx.Limits(
            max_connections=20,
            max_keepalive_connections=10,
            keepalive_expiry=30.0
        )

        self._client = httpx.AsyncClient(
            headers=headers,
            timeout=timeout,
            limits=limits,
            http2=True,  # Enable HTTP/2 for better performance
        )

    async def _rate_limit(self) -> None:
        """Apply rate limiting to API requests.

        This method ensures we don't exceed the configured rate limit
        by tracking request timestamps and adding delays when necessary.
        """
        async with self._rate_limit_lock:
            now = time.time()

            # Remove requests older than 1 minute
            cutoff = now - 60
            self._request_times = [t for t in self._request_times if t > cutoff]

            # Check if we're at the rate limit
            if len(self._request_times) >= self.rate_limit:
                # Calculate delay needed
                oldest_request = min(self._request_times)
                delay = 60 - (now - oldest_request) + 0.1  # Add small buffer

                if delay > 0:
                    logger.debug(f"Rate limiting: waiting {delay:.2f}s")
                    await asyncio.sleep(delay)

            # Record this request
            self._request_times.append(now)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    async def _make_embedding_request(self, texts: list[str]) -> list[list[float]]:
        """Make an embedding request to OpenAI API with retries.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors

        Raises:
            EmbeddingError: If the API request fails
        """
        await self._rate_limit()

        payload = {
            "model": self.model,
            "input": texts,
            "encoding_format": "float"
        }

        try:
            start_time = time.time()

            response = await self._client.post(
                "https://api.openai.com/v1/embeddings",
                json=payload
            )

            request_time = (time.time() - start_time) * 1000
            logger.debug(f"OpenAI API request took {request_time:.2f}ms for {len(texts)} texts")

            response.raise_for_status()
            result = response.json()

            # Extract embeddings in correct order
            embeddings = []
            for i in range(len(texts)):
                embedding_data = next(
                    (item for item in result["data"] if item["index"] == i),
                    None
                )
                if embedding_data is None:
                    raise EmbeddingError(f"Missing embedding for text at index {i}")

                embeddings.append(embedding_data["embedding"])

            # Log usage information
            usage = result.get("usage", {})
            total_tokens = usage.get("total_tokens", 0)
            logger.debug(f"OpenAI embedding request used {total_tokens} tokens")

            return embeddings

        except httpx.HTTPStatusError as e:
            error_msg = f"OpenAI API error {e.response.status_code}: {e.response.text}"
            logger.error(error_msg)

            # Handle specific error codes
            if e.response.status_code == 401:
                raise EmbeddingError("Invalid OpenAI API key") from e
            elif e.response.status_code == 429:
                raise EmbeddingError("OpenAI API rate limit exceeded") from e
            elif e.response.status_code == 400:
                raise EmbeddingError(f"Invalid request to OpenAI API: {e.response.text}") from e
            else:
                raise EmbeddingError(error_msg) from e

        except httpx.RequestError as e:
            error_msg = f"Network error communicating with OpenAI API: {e}"
            logger.error(error_msg)
            raise EmbeddingError(error_msg) from e

        except Exception as e:
            error_msg = f"Unexpected error in OpenAI embedding request: {e}"
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

        This method automatically splits large batches into smaller chunks
        to stay within API limits and processes them concurrently when possible.

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

        # Split into batches
        batches = [
            filtered_texts[i:i + self.batch_size]
            for i in range(0, len(filtered_texts), self.batch_size)
        ]

        logger.debug(f"Embedding {len(filtered_texts)} texts in {len(batches)} batches")

        # Process batches
        if len(batches) == 1:
            # Single batch - process directly
            embeddings = await self._make_embedding_request(batches[0])
        else:
            # Multiple batches - process with controlled concurrency
            semaphore = asyncio.Semaphore(3)  # Limit concurrent requests

            async def process_batch(batch):
                async with semaphore:
                    return await self._make_embedding_request(batch)

            batch_tasks = [process_batch(batch) for batch in batches]
            batch_results = await asyncio.gather(*batch_tasks)

            # Flatten results
            embeddings = []
            for batch_embeddings in batch_results:
                embeddings.extend(batch_embeddings)

        # Create result array with correct size, filling empty texts with None
        result = [None] * len(texts)
        for i, embedding in enumerate(embeddings):
            original_index = text_indices[i]
            result[original_index] = embedding

        # Replace None with empty embeddings for consistency
        dimension = self.get_embedding_dimension_sync()
        for i, embedding in enumerate(result):
            if embedding is None:
                result[i] = [0.0] * dimension
                logger.warning(f"Text at index {i} was empty, using zero embedding")

        return result

    async def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this backend.

        Returns:
            Integer dimension of embedding vectors
        """
        return self._model_dimensions.get(self.model, 1536)

    def get_embedding_dimension_sync(self) -> int:
        """Synchronous version of get_embedding_dimension for internal use."""
        return self._model_dimensions.get(self.model, 1536)

    async def close(self) -> None:
        """Clean up resources used by the backend."""
        if self._client:
            await self._client.aclose()
            self._client = None
            logger.debug("Closed OpenAI embedding backend HTTP client")

    def __del__(self):
        """Cleanup when object is destroyed."""
        if self._client and not self._client.is_closed:
            logger.warning("OpenAI embedding backend was not properly closed")

    async def get_usage_stats(self) -> dict[str, Any]:
        """Get usage statistics for monitoring.

        Returns:
            Dictionary with usage statistics
        """
        return {
            "backend_type": "openai",
            "model": self.model,
            "batch_size": self.batch_size,
            "rate_limit": self.rate_limit,
            "recent_requests": len(self._request_times),
            "client_connected": self._client is not None and not self._client.is_closed,
            "embedding_dimension": await self.get_embedding_dimension(),
        }
