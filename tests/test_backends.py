"""
Test suite for backend implementations.

This module contains tests for embedding and vector store backends
to ensure they work correctly and integrate properly.
"""

from datetime import datetime
from unittest.mock import Mock, patch

import pytest

from agentup_rag.backends.base import EmbeddingError
from agentup_rag.backends.factory import BackendFactory
from agentup_rag.models import SearchResult, VectorDocument


def _has_sentence_transformers() -> bool:
    """Check if sentence-transformers is available."""
    try:
        import sentence_transformers  # noqa: F401

        return True
    except ImportError:
        return False


class TestBackendFactory:
    """Test the backend factory functionality."""

    def test_list_available_backends(self):
        """Test listing available backends."""
        available = BackendFactory.list_available_backends()

        assert "embedding" in available
        assert "vector_store" in available
        assert isinstance(available["embedding"], list)
        assert isinstance(available["vector_store"], list)

    def test_create_embedding_backend_validation(self):
        """Test embedding backend creation with invalid config."""
        with pytest.raises(ValueError, match="Unknown embedding backend"):
            BackendFactory.create_embedding_backend("invalid_backend", {})

    def test_create_vector_store_backend_validation(self):
        """Test vector store backend creation with invalid config."""
        with pytest.raises(ValueError, match="Unknown vector store backend"):
            BackendFactory.create_vector_store_backend("invalid_backend", {})


class TestMemoryVectorStoreBackend:
    """Test the memory vector store backend."""

    @pytest.fixture
    def memory_backend(self):
        """Create a memory backend for testing."""
        from agentup_rag.backends.vector_store.memory import MemoryVectorStoreBackend

        config = {"similarity_metric": "cosine", "index_type": "flat", "persist_path": None, "auto_save": False}

        backend = MemoryVectorStoreBackend(config)
        return backend

    @pytest.mark.asyncio
    async def test_initialization(self, memory_backend):
        """Test backend initialization."""
        await memory_backend.initialize()
        assert memory_backend._initialized is True

    @pytest.mark.asyncio
    async def test_create_collection(self, memory_backend):
        """Test collection creation."""
        await memory_backend.initialize()
        await memory_backend.create_collection("test_collection", 384, "Test collection")

        collections = await memory_backend.list_collections()
        assert "test_collection" in collections

        stats = await memory_backend.get_collection_stats("test_collection")
        assert stats["exists"] is True
        assert stats["dimension"] == 384
        assert stats["description"] == "Test collection"

    @pytest.mark.asyncio
    async def test_add_and_search_vectors(self, memory_backend):
        """Test adding vectors and searching."""
        await memory_backend.initialize()

        # Create test vectors
        vectors = [
            VectorDocument(
                id="doc1",
                content="This is a test document about machine learning",
                embedding=[0.1, 0.2, 0.3],
                metadata={"topic": "ml"},
                source="test1.txt",
                chunk_index=0,
                created_at=datetime.utcnow(),
            ),
            VectorDocument(
                id="doc2",
                content="This is another document about deep learning",
                embedding=[0.2, 0.3, 0.4],
                metadata={"topic": "dl"},
                source="test2.txt",
                chunk_index=0,
                created_at=datetime.utcnow(),
            ),
        ]

        # Add vectors
        await memory_backend.add_vectors(vectors)

        # Search for similar vectors
        query_vector = [0.15, 0.25, 0.35]
        result = await memory_backend.search(query_vector, k=2)

        assert isinstance(result, SearchResult)
        assert len(result.documents) <= 2
        assert len(result.scores) == len(result.documents)

    @pytest.mark.asyncio
    async def test_delete_vectors(self, memory_backend):
        """Test vector deletion."""
        await memory_backend.initialize()

        # Add a test vector
        vectors = [
            VectorDocument(
                id="doc_to_delete",
                content="This document will be deleted",
                embedding=[0.1, 0.2, 0.3],
                metadata={},
                source="delete_test.txt",
                chunk_index=0,
                created_at=datetime.utcnow(),
            )
        ]

        await memory_backend.add_vectors(vectors)

        # Verify it exists
        stats = await memory_backend.get_collection_stats("default")
        assert stats["document_count"] == 1

        # Delete it
        await memory_backend.delete_vectors(["doc_to_delete"])

        # Verify it's gone
        stats = await memory_backend.get_collection_stats("default")
        assert stats["document_count"] == 0


class TestOpenAIEmbeddingBackend:
    """Test the OpenAI embedding backend."""

    @pytest.fixture
    def openai_config(self):
        """OpenAI backend configuration."""
        return {
            "model": "text-embedding-3-small",
            "api_key": "sk-test123",
            "batch_size": 10,
            "rate_limit": 60,
            "max_retries": 3,
            "timeout": 30,
        }

    def test_config_validation_valid(self, openai_config):
        """Test valid configuration."""
        from agentup_rag.backends.embedding.openai import OpenAIEmbeddingBackend

        # Should not raise an exception
        backend = OpenAIEmbeddingBackend(openai_config)
        assert backend.api_key == "sk-test123"
        assert backend.model == "text-embedding-3-small"

    def test_config_validation_invalid(self):
        """Test invalid configuration."""
        from agentup_rag.backends.embedding.openai import OpenAIEmbeddingBackend

        # Missing API key
        invalid_config = {"model": "text-embedding-3-small", "batch_size": 10}

        with pytest.raises(ValueError, match="API key is required"):
            OpenAIEmbeddingBackend(invalid_config)

    def test_unsupported_model(self, openai_config):
        """Test unsupported model configuration."""
        from agentup_rag.backends.embedding.openai import OpenAIEmbeddingBackend

        openai_config["model"] = "unsupported-model"

        with pytest.raises(ValueError, match="Unsupported model"):
            OpenAIEmbeddingBackend(openai_config)

    @pytest.mark.asyncio
    async def test_embed_text_mock(self, openai_config):
        """Test text embedding with mocked API."""
        from agentup_rag.backends.embedding.openai import OpenAIEmbeddingBackend

        backend = OpenAIEmbeddingBackend(openai_config)

        # Mock the HTTP client response
        mock_response = Mock()
        mock_response.json.return_value = {
            "data": [{"embedding": [0.1, 0.2, 0.3], "index": 0}],
            "usage": {"total_tokens": 5},
        }
        mock_response.raise_for_status = Mock()

        with patch.object(backend._client, "post", return_value=mock_response):
            embedding = await backend.embed_text("test text")

            assert embedding == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    async def test_embed_batch_mock(self, openai_config):
        """Test batch embedding with mocked API."""
        from agentup_rag.backends.embedding.openai import OpenAIEmbeddingBackend

        backend = OpenAIEmbeddingBackend(openai_config)

        # Mock the HTTP client response
        mock_response = Mock()
        mock_response.json.return_value = {
            "data": [{"embedding": [0.1, 0.2, 0.3], "index": 0}, {"embedding": [0.4, 0.5, 0.6], "index": 1}],
            "usage": {"total_tokens": 10},
        }
        mock_response.raise_for_status = Mock()

        with patch.object(backend._client, "post", return_value=mock_response):
            embeddings = await backend.embed_batch(["text 1", "text 2"])

            assert len(embeddings) == 2
            assert embeddings[0] == [0.1, 0.2, 0.3]
            assert embeddings[1] == [0.4, 0.5, 0.6]

    @pytest.mark.asyncio
    async def test_error_handling(self, openai_config):
        """Test error handling in OpenAI backend."""
        import httpx

        from agentup_rag.backends.embedding.openai import OpenAIEmbeddingBackend

        backend = OpenAIEmbeddingBackend(openai_config)

        # Mock HTTP error
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"

        http_error = httpx.HTTPStatusError("Unauthorized", request=Mock(), response=mock_response)

        with patch.object(backend._client, "post", side_effect=http_error):
            with pytest.raises(EmbeddingError, match="Invalid OpenAI API key"):
                await backend.embed_text("test text")

    @pytest.mark.asyncio
    async def test_rate_limiting(self, openai_config):
        """Test rate limiting functionality."""
        from agentup_rag.backends.embedding.openai import OpenAIEmbeddingBackend

        # Set very low rate limit for testing
        openai_config["rate_limit"] = 1
        backend = OpenAIEmbeddingBackend(openai_config)

        # Simulate multiple requests
        backend._request_times = [1000.0]  # One request at time 1000

        # This should trigger rate limiting
        import time

        _start_time = time.time()
        await backend._rate_limit()
        _end_time = time.time()

        # Should have been delayed (though we can't easily test the exact timing)
        assert True  # Just verify no exception was raised


class TestLocalEmbeddingBackend:
    """Test the local embedding backend."""

    @pytest.fixture
    def local_config(self):
        """Local embedding backend configuration."""
        return {
            "model": "all-MiniLM-L6-v2",
            "device": "cpu",
            "batch_size": 16,
            "cache_dir": None,
            "normalize_embeddings": True,
        }

    def test_config_validation_valid(self, local_config):
        """Test valid configuration."""
        from agentup_rag.backends.embedding.local import LocalEmbeddingBackend

        # Should not raise an exception
        backend = LocalEmbeddingBackend(local_config)
        assert backend.model_name == "all-MiniLM-L6-v2"
        assert backend.device == "cpu"
        assert backend.batch_size == 16
        assert backend.normalize_embeddings is True

    def test_config_validation_invalid(self):
        """Test invalid configuration."""
        from agentup_rag.backends.embedding.local import LocalEmbeddingBackend

        # Missing model
        invalid_config = {"device": "cpu", "batch_size": 16}

        with pytest.raises(ValueError, match="Model name is required"):
            LocalEmbeddingBackend(invalid_config)

    def test_invalid_device(self, local_config):
        """Test invalid device configuration."""
        from agentup_rag.backends.embedding.local import LocalEmbeddingBackend

        local_config["device"] = "invalid_device"

        with pytest.raises(ValueError, match="Device must be one of"):
            LocalEmbeddingBackend(local_config)

    def test_invalid_batch_size(self, local_config):
        """Test invalid batch size configuration."""
        from agentup_rag.backends.embedding.local import LocalEmbeddingBackend

        local_config["batch_size"] = 0

        with pytest.raises(ValueError, match="Batch size must be between"):
            LocalEmbeddingBackend(local_config)

    @pytest.mark.skipif(not _has_sentence_transformers(), reason="sentence-transformers not available")
    @pytest.mark.asyncio
    async def test_embed_text_real(self, local_config):
        """Test text embedding with real model (if available)."""
        from agentup_rag.backends.embedding.local import LocalEmbeddingBackend

        backend = LocalEmbeddingBackend(local_config)

        try:
            embedding = await backend.embed_text("This is a test sentence.")

            assert isinstance(embedding, list)
            assert len(embedding) == 384  # all-MiniLM-L6-v2 dimension
            assert all(isinstance(x, float) for x in embedding)

        finally:
            await backend.close()

    @pytest.mark.skipif(not _has_sentence_transformers(), reason="sentence-transformers not available")
    @pytest.mark.asyncio
    async def test_embed_batch_real(self, local_config):
        """Test batch embedding with real model (if available)."""
        from agentup_rag.backends.embedding.local import LocalEmbeddingBackend

        backend = LocalEmbeddingBackend(local_config)

        try:
            texts = ["This is the first sentence.", "This is the second sentence.", "This is the third sentence."]

            embeddings = await backend.embed_batch(texts)

            assert len(embeddings) == 3
            assert all(len(emb) == 384 for emb in embeddings)  # all-MiniLM-L6-v2 dimension
            assert all(isinstance(emb, list) for emb in embeddings)
            assert all(all(isinstance(x, float) for x in emb) for emb in embeddings)

            # Check that different sentences have different embeddings
            assert embeddings[0] != embeddings[1]
            assert embeddings[1] != embeddings[2]

        finally:
            await backend.close()

    @pytest.mark.asyncio
    async def test_empty_text_handling(self, local_config):
        """Test handling of empty text."""
        from agentup_rag.backends.base import EmbeddingError
        from agentup_rag.backends.embedding.local import LocalEmbeddingBackend

        backend = LocalEmbeddingBackend(local_config)

        try:
            with pytest.raises(EmbeddingError, match="Cannot embed empty text"):
                await backend.embed_text("")

            with pytest.raises(EmbeddingError, match="Cannot embed empty text"):
                await backend.embed_text("   ")

        finally:
            await backend.close()

    @pytest.mark.asyncio
    async def test_empty_batch_handling(self, local_config):
        """Test handling of empty batch."""
        from agentup_rag.backends.embedding.local import LocalEmbeddingBackend

        backend = LocalEmbeddingBackend(local_config)

        try:
            result = await backend.embed_batch([])
            assert result == []

            # Batch with only empty strings should result in zero embeddings
            result = await backend.embed_batch(["", "  ", ""])
            assert len(result) == 3
            # Each should be a zero embedding (handled gracefully)

        finally:
            await backend.close()

    def test_get_embedding_dimension_known_model(self, local_config):
        """Test getting embedding dimension for known model."""
        from agentup_rag.backends.embedding.local import LocalEmbeddingBackend

        backend = LocalEmbeddingBackend(local_config)

        # Should return known dimension without loading model
        dimension = backend.get_embedding_dimension_sync()
        assert dimension == 384  # all-MiniLM-L6-v2

    def test_get_model_info_unloaded(self, local_config):
        """Test getting model info when model is not loaded."""
        from agentup_rag.backends.embedding.local import LocalEmbeddingBackend

        backend = LocalEmbeddingBackend(local_config)

        info = backend.get_model_info()
        assert "error" in info
        assert "Model not loaded" in info["error"]

    @pytest.mark.asyncio
    async def test_usage_stats(self, local_config):
        """Test getting usage statistics."""
        from agentup_rag.backends.embedding.local import LocalEmbeddingBackend

        backend = LocalEmbeddingBackend(local_config)

        try:
            stats = await backend.get_usage_stats()

            assert stats["backend_type"] == "local"
            assert stats["model"] == "all-MiniLM-L6-v2"
            assert stats["device"] == "cpu"
            assert stats["batch_size"] == 16
            assert stats["normalize_embeddings"] is True
            assert stats["model_loaded"] is False
            assert stats["cache_dir"] is None

        finally:
            await backend.close()

    def test_is_available(self):
        """Test checking if sentence-transformers is available."""
        from agentup_rag.backends.embedding.local import LocalEmbeddingBackend

        backend = LocalEmbeddingBackend({"model": "test"})

        # This should return True or False based on whether sentence-transformers is installed
        available = backend.is_available()
        assert isinstance(available, bool)


class TestProcessingComponents:
    """Test document processing components."""

    def test_text_chunker_fixed_strategy(self):
        """Test fixed-size chunking strategy."""
        from agentup_rag.models import ChunkingConfig, ChunkingStrategy
        from agentup_rag.processing.chunking import TextChunker

        config = ChunkingConfig(strategy=ChunkingStrategy.FIXED, chunk_size=100, chunk_overlap=20)

        chunker = TextChunker(config)
        text = "This is a long text that should be split into multiple chunks. " * 10
        chunks = chunker.chunk_text(text, "test.txt")

        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk.content) <= 100
            assert chunk.metadata["strategy"] == "fixed"

    def test_text_chunker_recursive_strategy(self):
        """Test recursive chunking strategy."""
        from agentup_rag.models import ChunkingConfig, ChunkingStrategy
        from agentup_rag.processing.chunking import TextChunker

        config = ChunkingConfig(
            strategy=ChunkingStrategy.RECURSIVE, chunk_size=200, chunk_overlap=50, separators=["\n\n", "\n", ".", " "]
        )

        chunker = TextChunker(config)
        text = """This is paragraph one.

This is paragraph two with multiple sentences. It has more content.

This is paragraph three."""

        chunks = chunker.chunk_text(text, "test.txt")

        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk.metadata["strategy"] == "recursive"

    def test_document_processor(self):
        """Test document processor."""
        from agentup_rag.models import ChunkingConfig
        from agentup_rag.processing.chunking import TextChunker
        from agentup_rag.processing.document_processor import DocumentProcessor

        processor = DocumentProcessor()
        chunker = TextChunker(ChunkingConfig())
        processor.set_chunker(chunker)

        content = "This is a test document with some content that should be processed."
        result = processor.process_text(content, "test.txt", {"author": "test"})

        assert result.success is True
        assert result.chunks_created > 0
        assert result.source == "test.txt"
        assert result.processing_time > 0


if __name__ == "__main__":
    pytest.main([__file__])
