"""
Test suite for the RAG plugin.

This module contains unit and integration tests for the RAG plugin
to ensure all components work correctly.
"""

from unittest.mock import AsyncMock, Mock

import pytest

from agentup_rag.plugin import RAGPlugin


class TestRAGPlugin:
    """Test the main RAG plugin functionality."""

    def test_plugin_registration(self):
        """Test that the plugin registers correctly."""
        plugin = RAGPlugin()
        capabilities = plugin.register_capability()

        # Should return a list of capabilities
        assert isinstance(capabilities, list)
        assert len(capabilities) == 4  # index_document, semantic_search, ask_documents, list_collections

        # Check first capability (index_document)
        first_capability = capabilities[0]
        assert first_capability.id == "index_document"
        assert first_capability.name == "Index Document"
        assert first_capability.version == "0.1.0"
        assert "documents to the vector index" in first_capability.description.lower()

    def test_config_validation_valid(self):
        """Test configuration validation with valid config."""
        plugin = RAGPlugin()

        config = {
            "embedding_backend": "openai",
            "embedding_config": {"model": "text-embedding-3-small", "api_key": "sk-test123"},
            "vector_backend": "memory",
            "vector_config": {"similarity_metric": "cosine", "index_type": "flat"},
        }

        result = plugin.validate_config(config)
        assert result.valid is True
        assert len(result.errors) == 0

    def test_config_validation_invalid(self):
        """Test configuration validation with invalid config."""
        plugin = RAGPlugin()

        # Missing required fields
        config = {
            "embedding_backend": "openai",
        }

        result = plugin.validate_config(config)
        assert result.valid is False
        assert len(result.errors) > 0

    def test_can_handle_task_high_confidence(self):
        """Test task routing with high confidence queries."""
        plugin = RAGPlugin()

        # Mock context with search-related query
        context = Mock()
        context.task = Mock()
        context.task.history = [Mock()]
        context.task.history[0].parts = [Mock()]
        context.task.history[0].parts[0].text = "search for documents about machine learning"

        confidence = plugin.can_handle_task(context)
        assert confidence >= 0.8

    def test_can_handle_task_low_confidence(self):
        """Test task routing with low confidence queries."""
        plugin = RAGPlugin()

        # Mock context with unrelated query
        context = Mock()
        context.task = Mock()
        context.task.history = [Mock()]
        context.task.history[0].parts = [Mock()]
        context.task.history[0].parts[0].text = "what's the weather like today?"

        confidence = plugin.can_handle_task(context)
        assert confidence < 0.5

    def test_get_ai_functions(self):
        """Test AI function registration."""
        plugin = RAGPlugin()
        ai_functions = plugin.get_ai_functions()

        assert len(ai_functions) == 4  # index_document, semantic_search, ask_documents, list_collections

        function_names = [f.name for f in ai_functions]
        assert "index_document" in function_names
        assert "semantic_search" in function_names
        assert "ask_documents" in function_names
        assert "list_collections" in function_names

    @pytest.mark.asyncio
    async def test_list_collections_function(self):
        """Test the list_collections AI function."""
        plugin = RAGPlugin()

        # Mock vector backend
        mock_backend = AsyncMock()
        mock_backend.list_collections.return_value = ["collection1", "collection2"]
        plugin.vector_backend = mock_backend

        # Mock context
        task = Mock()
        context = Mock()
        context.metadata = {"parameters": {}}

        result = await plugin._handle_list_collections_function(task, context)

        assert result.success is True
        assert "collection1" in result.content
        assert "collection2" in result.content
        assert result.metadata["collections"] == ["collection1", "collection2"]

    @pytest.mark.asyncio
    async def test_list_collections_function_empty(self):
        """Test list_collections with no collections."""
        plugin = RAGPlugin()

        # Mock vector backend
        mock_backend = AsyncMock()
        mock_backend.list_collections.return_value = []
        plugin.vector_backend = mock_backend

        # Mock context
        task = Mock()
        context = Mock()
        context.metadata = {"parameters": {}}

        result = await plugin._handle_list_collections_function(task, context)

        assert result.success is True
        assert "No collections found" in result.content
        assert result.metadata["collections"] == []


class TestRAGPluginIntegration:
    """Integration tests for the RAG plugin with real components."""

    def setup_method(self):
        """Set up test fixtures."""
        self.plugin = RAGPlugin()

        # Mock configuration
        self.test_config = {
            "embedding_backend": "openai",
            "embedding_config": {"model": "text-embedding-3-small", "api_key": "sk-test123", "batch_size": 10},
            "vector_backend": "memory",
            "vector_config": {"similarity_metric": "cosine", "index_type": "flat"},
            "chunking": {"strategy": "recursive", "chunk_size": 1000, "chunk_overlap": 200},
            "search": {"default_k": 5, "max_k": 50},
        }

    @pytest.mark.asyncio
    async def test_integration_workflow(self):
        """Test a complete RAG workflow."""
        # This is a mock integration test that would test the full pipeline
        # In a real implementation, you'd set up real backends and test the flow

        # Mock services
        services = {
            "http_client": AsyncMock(),
            "cache": AsyncMock(),
        }

        self.plugin.configure_services(services)

        # Test configuration validation
        validation_result = self.plugin.validate_config(self.test_config)
        assert validation_result.valid is True

        # Test that AI functions are available
        ai_functions = self.plugin.get_ai_functions()
        assert len(ai_functions) > 0

    def test_middleware_config(self):
        """Test middleware configuration."""
        plugin = RAGPlugin()
        middleware_config = plugin.get_middleware_config()

        assert len(middleware_config) > 0

        # Check for expected middleware types
        middleware_types = [config["type"] for config in middleware_config]
        assert "rate_limit" in middleware_types
        assert "cache" in middleware_types
        assert "retry" in middleware_types

    def test_state_schema(self):
        """Test state schema definition."""
        plugin = RAGPlugin()
        state_schema = plugin.get_state_schema()

        assert "type" in state_schema
        assert state_schema["type"] == "object"
        assert "properties" in state_schema

        # Check for expected state properties
        properties = state_schema["properties"]
        assert "last_search_query" in properties
        assert "search_count" in properties
        assert "index_count" in properties


@pytest.mark.asyncio
async def test_plugin_lifecycle():
    """Test the plugin lifecycle (initialization and cleanup)."""
    plugin = RAGPlugin()

    # Mock backends
    embedding_backend = AsyncMock()
    vector_backend = AsyncMock()
    http_client = AsyncMock()

    plugin.embedding_backend = embedding_backend
    plugin.vector_backend = vector_backend
    plugin.http_client = http_client

    # Test cleanup
    await plugin.cleanup()

    # Verify cleanup was called on all components
    embedding_backend.close.assert_called_once()
    vector_backend.close.assert_called_once()
    http_client.aclose.assert_called_once()


class TestRAGPluginErrorHandling:
    """Test error handling in the RAG plugin."""

    @pytest.mark.asyncio
    async def test_function_with_uninitialized_backends(self):
        """Test AI functions with uninitialized backends."""
        plugin = RAGPlugin()

        # Ensure backends are not initialized
        plugin.embedding_backend = None
        plugin.vector_backend = None

        # Mock context - plugin now gets params from task.metadata
        task = Mock()
        task.metadata = {"content": "test document", "source": "test.txt"}
        context = Mock()
        context.config = {}

        result = await plugin._handle_index_document_function(task, context)

        assert result.success is False
        assert "no configuration available" in result.content.lower()
        assert result.error == "NO_CONFIG"

    @pytest.mark.asyncio
    async def test_empty_content_handling(self):
        """Test handling of empty content."""
        plugin = RAGPlugin()

        # Mock context with empty content
        task = Mock()
        task.metadata = {"content": "", "source": "test.txt"}
        context = Mock()
        context.config = {}

        result = await plugin._handle_index_document_function(task, context)

        assert result.success is False
        assert "empty" in result.content.lower()
        assert result.error == "EMPTY_CONTENT"

    @pytest.mark.asyncio
    async def test_empty_query_handling(self):
        """Test handling of empty search queries."""
        plugin = RAGPlugin()

        # Mock context with empty query
        task = Mock()
        task.metadata = {"query": ""}
        context = Mock()
        context.config = {}

        # Mock plugin config to avoid comparison errors
        plugin.config = Mock()
        plugin.config.default_collection = "default"
        plugin.config.max_k = 100

        result = await plugin._handle_semantic_search_function(task, context)

        assert result.success is False
        assert "empty" in result.content.lower()
        assert result.error == "EMPTY_QUERY"


if __name__ == "__main__":
    pytest.main([__file__])
