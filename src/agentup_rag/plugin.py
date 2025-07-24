"""
Main RAG plugin implementation for AgentUp.

This module contains the primary plugin class that integrates with AgentUp's
hook system to provide RAG (Retrieval-Augmented Generation) capabilities.
"""

import logging
import time
from datetime import datetime
from typing import Any

import pluggy
from agent.plugins import (
    AIFunction,
    CapabilityContext,
    CapabilityInfo,
    CapabilityResult,
    CapabilityType,
    ValidationResult,
)

from .backends.base import EmbeddingBackend, VectorStoreBackend
from .backends.factory import BackendFactory
from .models import (
    RAGConfig,
    VectorDocument,
)
from .processing import DocumentProcessor, TextChunker

hookimpl = pluggy.HookimplMarker("agentup")
logger = logging.getLogger(__name__)


class RAGPlugin:
    """Main RAG plugin class implementing AgentUp hook interface.

    This plugin provides comprehensive RAG capabilities including:
    - Document indexing with multiple chunking strategies
    - Semantic search with configurable backends
    - RAG-style question answering
    - Collection management
    - AI function integration for LLM use
    """

    def __init__(self):
        """Initialize the RAG plugin."""
        self.name = "agentup-rag"
        self.version = "0.1.0"

        # Backend instances
        self.embedding_backend: EmbeddingBackend | None = None
        self.vector_backend: VectorStoreBackend | None = None

        # Configuration
        self.config: RAGConfig | None = None

        # Processing components
        self.document_processor: DocumentProcessor | None = None
        self.text_chunker: TextChunker | None = None

        # Services injected by AgentUp
        self.http_client = None
        self.cache = None
        self.services = {}

        logger.info(f"Initialized RAG plugin v{self.version}")

    @hookimpl
    def register_capability(self) -> list[CapabilityInfo]:
        """Register the RAG capabilities with AgentUp.

        Returns:
            list of CapabilityInfo describing the plugin's capabilities
        """
        return [
            CapabilityInfo(
                id="index_document",
                name="Index Document",
                version=self.version,
                description="Add documents to the vector index for semantic search",
                plugin_name="rag",
                capabilities=[CapabilityType.AI_FUNCTION],
                tags=["rag", "index", "documents"],
                required_scopes=["rag:access"],
                config_schema=self._get_config_schema(),
            ),
            CapabilityInfo(
                id="semantic_search",
                name="Semantic Search",
                version=self.version,
                description="Search for semantically similar content",
                plugin_name="rag",
                capabilities=[CapabilityType.AI_FUNCTION],
                tags=["rag", "search", "semantic"],
                required_scopes=["rag:access"],
                config_schema=self._get_config_schema(),
            ),
            CapabilityInfo(
                id="ask_documents",
                name="Ask Documents",
                version=self.version,
                description="Ask questions and get answers based on indexed documents",
                plugin_name="rag",
                capabilities=[CapabilityType.AI_FUNCTION],
                tags=["rag", "qa", "retrieval"],
                required_scopes=["rag:access"],
                config_schema=self._get_config_schema(),
            ),
            CapabilityInfo(
                id="list_collections",
                name="list Collections",
                version=self.version,
                description="list available document collections",
                plugin_name="rag",
                capabilities=[CapabilityType.AI_FUNCTION],
                tags=["rag", "collections"],
                required_scopes=["rag:access"],
                config_schema=self._get_config_schema(),
            ),
        ]

    def _get_system_prompt(self) -> str:
        """Get the system prompt for RAG operations.

        Returns:
            System prompt string
        """
        return """You are a helpful AI assistant with access to a semantic search system.
        When users ask questions, you can search through indexed documents to find
        relevant information and provide accurate, well-sourced answers.

        Always cite your sources when providing information from documents.
        If you cannot find relevant information, say so clearly.

        Available capabilities:
        - index_document: Add new documents to the search index
        - semantic_search: Find similar content using semantic search
        - ask_documents: Answer questions using retrieval-augmented generation
        - list_collections: Show available document collections
        - delete_documents: Remove documents from the index
        """

    def _get_config_schema(self) -> dict[str, Any]:
        """Get the configuration schema for the plugin.

        Returns:
            JSON schema for plugin configuration
        """
        return {
            "type": "object",
            "properties": {
                "embedding_backend": {
                    "type": "string",
                    "enum": ["openai", "local"],
                    "description": "Embedding backend to use",
                },
                "embedding_config": {"type": "object", "description": "Embedding backend configuration"},
                "vector_backend": {
                    "type": "string",
                    "enum": ["memory", "chroma", "pinecone", "weaviate"],
                    "description": "Vector store backend to use",
                },
                "vector_config": {"type": "object", "description": "Vector store backend configuration"},
                "chunking": {
                    "type": "object",
                    "properties": {
                        "strategy": {
                            "type": "string",
                            "enum": ["fixed", "recursive", "semantic"],
                            "default": "recursive",
                        },
                        "chunk_size": {"type": "integer", "default": 1000, "minimum": 100},
                        "chunk_overlap": {"type": "integer", "default": 200, "minimum": 0},
                    },
                },
                "search": {
                    "type": "object",
                    "properties": {
                        "default_k": {"type": "integer", "default": 5, "minimum": 1},
                        "max_k": {"type": "integer", "default": 100, "minimum": 1},
                    },
                },
            },
            "required": ["embedding_backend", "embedding_config", "vector_backend", "vector_config"],
        }

    @hookimpl
    def validate_config(self, config: dict[str, Any]) -> ValidationResult:
        """Validate the plugin configuration.

        Args:
            config: Configuration dictionary to validate

        Returns:
            ValidationResult indicating if config is valid
        """
        errors = []
        warnings = []
        suggestions = []

        try:
            # Validate using Pydantic model
            rag_config = RAGConfig(**config)

            # Store the validated configuration for later use
            self.config = rag_config
            logger.info("RAG plugin configuration validated and stored")

            # Additional validations
            if rag_config.embedding_backend == "openai":
                api_key = rag_config.embedding_config.api_key
                if not api_key or not api_key.strip():
                    errors.append("OpenAI API key is required")
                elif not api_key.startswith("sk-"):
                    warnings.append("OpenAI API key should start with 'sk-'")

            if rag_config.vector_backend == "pinecone":
                if not rag_config.vector_config.api_key:
                    errors.append("Pinecone API key is required")
                if not rag_config.vector_config.environment:
                    errors.append("Pinecone environment is required")

            # Performance suggestions
            if rag_config.chunking.chunk_size > 2000:
                suggestions.append("Large chunk sizes may impact search quality")

            if rag_config.default_k > 20:
                suggestions.append("Large k values may impact performance")

        except Exception as e:
            errors.append(f"Configuration validation failed: {str(e)}")

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions,
        )

    @hookimpl
    def configure_services(self, services: dict[str, Any]) -> None:
        """Configure services for the plugin.

        Args:
            services: dictionary of available services
        """
        self.services = services

        # Get HTTP client for API calls
        self.http_client = services.get("http_client")
        if not self.http_client:
            import httpx

            self.http_client = httpx.AsyncClient()

        # Get cache service if available
        self.cache = services.get("cache")

        logger.debug("Configured RAG plugin services")

    @hookimpl
    def can_handle_task(self, context: CapabilityContext) -> float:
        """Determine if this plugin can handle a task.

        Args:
            context: Task context

        Returns:
            Confidence score (0.0 - 1.0)
        """
        user_input = self._extract_user_input(context).lower()

        # High confidence keywords
        rag_keywords = {
            "search": 1.0,
            "find": 0.9,
            "document": 0.9,
            "index": 0.9,
            "retrieval": 1.0,
            "semantic": 1.0,
            "similar": 0.8,
            "lookup": 0.8,
            "query": 0.8,
            "ask": 0.7,
            "question": 0.7,
        }

        # Question patterns that suggest RAG use
        rag_patterns = [
            r"what (?:is|are|does|do)",
            r"how (?:to|do|does)",
            r"where (?:is|are|can)",
            r"when (?:is|are|was|were)",
            r"why (?:is|are|does|do)",
            r"find (?:documents?|information|data)",
            r"search (?:for|in|through)",
            r"tell me about",
            r"explain",
            r"describe",
        ]

        confidence = 0.0

        # Check keyword matches
        for keyword, score in rag_keywords.items():
            if keyword in user_input:
                confidence = max(confidence, score)

        # Check pattern matches
        import re

        for pattern in rag_patterns:
            if re.search(pattern, user_input):
                confidence = max(confidence, 0.8)

        # Boost confidence for explicit document/search references
        if any(word in user_input for word in ["document", "search", "find", "lookup"]):
            confidence = min(confidence + 0.2, 1.0)

        return confidence

    @hookimpl
    def execute_skill(self, context: CapabilityContext) -> CapabilityResult:
        """Execute the RAG skill logic.

        Args:
            context: Skill execution context

        Returns:
            CapabilityResult with the execution outcome
        """
        try:
            # Initialize if not already done
            if not self.embedding_backend or not self.vector_backend:
                import asyncio

                asyncio.run(self._initialize_backends(context.config))

            # Extract user input and determine intent
            user_input = self._extract_user_input(context)
            intent = self._determine_intent(user_input)

            # Route to appropriate handler
            if intent == "index":
                return self._handle_index_request(context, user_input)
            elif intent == "search":
                return self._handle_search_request(context, user_input)
            elif intent == "ask":
                return self._handle_ask_request(context, user_input)
            else:
                # Default to search
                return self._handle_search_request(context, user_input)

        except Exception as e:
            logger.error(f"RAG skill execution failed: {e}", exc_info=True)
            return CapabilityResult(
                content=f"Sorry, I encountered an error: {str(e)}",
                success=False,
                error=str(e),
            )

    async def _initialize_backends(self, config: dict[str, Any]) -> None:
        """Initialize embedding and vector store backends.

        Args:
            config: Plugin configuration
        """
        try:
            self.config = RAGConfig(**config)

            # Initialize embedding backend
            self.embedding_backend = BackendFactory.create_embedding_backend(
                self.config.embedding_backend, self.config.embedding_config.dict()
            )

            # Initialize vector store backend
            self.vector_backend = BackendFactory.create_vector_store_backend(
                self.config.vector_backend, self.config.vector_config.dict()
            )

            # Initialize vector backend (properly await it)
            await self.vector_backend.initialize()

            # Initialize processing components
            from .processing import DocumentProcessor, TextChunker

            self.document_processor = DocumentProcessor()
            self.text_chunker = TextChunker(self.config.chunking)

            logger.info("RAG backends initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize RAG backends: {e}")
            raise

    def _extract_user_input(self, context: CapabilityContext) -> str:
        """Extract user input from the task context.

        Args:
            context: Skill context

        Returns:
            User input text
        """
        if hasattr(context.task, "history") and context.task.history:
            last_msg = context.task.history[-1]
            if hasattr(last_msg, "parts") and last_msg.parts:
                return last_msg.parts[0].text if hasattr(last_msg.parts[0], "text") else ""
        return ""

    def _determine_intent(self, user_input: str) -> str:
        """Determine user intent from input.

        Args:
            user_input: User input text

        Returns:
            Intent string ('index', 'search', 'ask')
        """
        user_input = user_input.lower()

        # Index intent keywords
        if any(word in user_input for word in ["index", "add", "store", "save", "upload"]):
            return "index"

        # Question/ask intent keywords
        if any(word in user_input for word in ["what", "how", "why", "when", "where", "explain", "tell me"]):
            return "ask"

        # Default to search
        return "search"

    def _handle_index_request(self, context: CapabilityContext, user_input: str) -> CapabilityResult:
        """Handle document indexing requests.

        Args:
            context: Skill context
            user_input: User input text

        Returns:
            CapabilityResult with indexing outcome
        """
        # This is a placeholder - in a real implementation, you'd need to
        # extract document content from the context or request it from the user
        return CapabilityResult(
            content="Document indexing is available through the index_document AI function. "
            "Please use that function to add documents to the search index.",
            success=True,
            metadata={"intent": "index"},
        )

    def _handle_search_request(self, context: CapabilityContext, user_input: str) -> CapabilityResult:
        """Handle search requests.

        Args:
            context: Skill context
            user_input: User input text

        Returns:
            CapabilityResult with search results
        """
        # This is a placeholder - actual search would be implemented via AI functions
        return CapabilityResult(
            content="Search functionality is available through the semantic_search AI function. "
            "Please use that function to search for relevant documents.",
            success=True,
            metadata={"intent": "search", "query": user_input},
        )

    def _handle_ask_request(self, context: CapabilityContext, user_input: str) -> CapabilityResult:
        """Handle question answering requests.

        Args:
            context: Skill context
            user_input: User input text

        Returns:
            CapabilityResult with answer
        """
        # This is a placeholder - actual QA would be implemented via AI functions
        return CapabilityResult(
            content="Question answering is available through the ask_documents AI function. "
            "Please use that function to get answers based on indexed documents.",
            success=True,
            metadata={"intent": "ask", "question": user_input},
        )

    @hookimpl
    def get_ai_functions(self, capability_id: str = None) -> list[AIFunction]:
        """Provide AI functions for LLM integration.

        Args:
            capability_id: Optional capability ID to filter functions for

        Returns:
            list of AI functions that can be called by LLMs
        """
        # Define all AI functions
        all_functions = [
            AIFunction(
                name="index_document",
                description="Add a document to the vector index for semantic search",
                parameters={
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "Document content to index"},
                        "source": {"type": "string", "description": "Source identifier (filename, URL, etc.)"},
                        "collection": {
                            "type": "string",
                            "description": "Collection name (optional)",
                            "default": "default",
                        },
                        "metadata": {
                            "type": "object",
                            "description": "Additional metadata to store",
                            "additionalProperties": True,
                        },
                    },
                    "required": ["content"],
                },
                handler=self._handle_index_document_function,
            ),
            AIFunction(
                name="semantic_search",
                description="Search for semantically similar content",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query text"},
                        "collection": {
                            "type": "string",
                            "description": "Collection to search in",
                            "default": "default",
                        },
                        "k": {
                            "type": "integer",
                            "description": "Number of results to return",
                            "minimum": 1,
                            "maximum": 100,
                            "default": 5,
                        },
                        "similarity_threshold": {
                            "type": "number",
                            "description": "Minimum similarity score",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "default": 0.0,
                        },
                        "filters": {"type": "object", "description": "Metadata filters", "additionalProperties": True},
                    },
                    "required": ["query"],
                },
                handler=self._handle_semantic_search_function,
            ),
            AIFunction(
                name="ask_documents",
                description="Ask a question and get an answer based on indexed documents",
                parameters={
                    "type": "object",
                    "properties": {
                        "question": {"type": "string", "description": "Question to answer"},
                        "collection": {
                            "type": "string",
                            "description": "Collection to search in",
                            "default": "default",
                        },
                        "max_context_length": {
                            "type": "integer",
                            "description": "Maximum context length for RAG",
                            "default": 4000,
                        },
                        "include_sources": {
                            "type": "boolean",
                            "description": "Include source references in response",
                            "default": True,
                        },
                    },
                    "required": ["question"],
                },
                handler=self._handle_ask_documents_function,
            ),
            AIFunction(
                name="list_collections",
                description="list available document collections",
                parameters={"type": "object", "properties": {}},
                handler=self._handle_list_collections_function,
            ),
        ]

        # Filter functions based on capability_id if provided
        if capability_id:
            # Map capability IDs to their corresponding function names
            capability_to_function = {
                "index_document": "index_document",
                "semantic_search": "semantic_search",
                "ask_documents": "ask_documents",
                "list_collections": "list_collections",
            }

            function_name = capability_to_function.get(capability_id)
            if function_name:
                return [f for f in all_functions if f.name == function_name]
            else:
                return []

        # Return all functions if no capability_id specified
        return all_functions

    async def _handle_index_document_function(self, task, context: CapabilityContext) -> CapabilityResult:
        """Handle the index_document AI function.

        Args:
            task: Task object
            context: Skill context with parameters

        Returns:
            CapabilityResult with indexing outcome
        """
        try:
            # Extract parameters from task metadata (where function arguments are placed)
            # First try the task metadata (new format), then fall back to context metadata (legacy)
            if hasattr(task, "metadata") and task.metadata:
                params = task.metadata
            else:
                params = context.metadata.get("parameters", {})

            content = params.get("content")
            source = params.get("source", "unknown")
            collection_name = params.get("collection", self.config.default_collection if self.config else "default")
            metadata = params.get("metadata", {})

            # Validate input
            if not content or len(content.strip()) == 0:
                return CapabilityResult(
                    content="Error: Document content cannot be empty", success=False, error="EMPTY_CONTENT"
                )

            # Initialize backends if needed
            if not self.embedding_backend or not self.vector_backend:
                try:
                    # Use context config if available, otherwise use stored config
                    config_to_use = context.config if context.config else (self.config.dict() if self.config else None)
                    if not config_to_use:
                        return CapabilityResult(
                            content="Error: No configuration available for RAG plugin", success=False, error="NO_CONFIG"
                        )
                    await self._initialize_backends(config_to_use)
                except Exception as e:
                    logger.error(f"Failed to initialize RAG backends: {e}")
                    return CapabilityResult(
                        content=f"Error: Failed to initialize RAG backends: {str(e)}",
                        success=False,
                        error="BACKEND_INIT_FAILED",
                    )

            start_time = time.time()

            # Process document
            processing_result = self.document_processor.process_text(content, source, metadata)

            if not processing_result.success:
                return CapabilityResult(
                    content=f"Error processing document: {processing_result.error}",
                    success=False,
                    error=processing_result.error,
                )

            # Get chunks from processing result
            chunks = self.text_chunker.chunk_text(content, source)

            # Generate embeddings for chunks
            texts = [chunk.content for chunk in chunks]
            embeddings = await self.embedding_backend.embed_batch(texts)

            # Create vector documents
            vector_docs = []
            for chunk, embedding in zip(chunks, embeddings, strict=False):
                vector_doc = VectorDocument(
                    id=VectorDocument.generate_id(chunk.content, source, chunk.chunk_index),
                    content=chunk.content,
                    embedding=embedding,
                    metadata={**chunk.metadata, **metadata},
                    source=source,
                    chunk_index=chunk.chunk_index,
                    created_at=datetime.utcnow(),
                )
                vector_docs.append(vector_doc)

            # Store in vector database
            await self.vector_backend.add_vectors(vector_docs, collection_name)

            processing_time = (time.time() - start_time) * 1000

            return CapabilityResult(
                content=f"Successfully indexed {len(vector_docs)} chunks from document '{source}' in collection '{collection_name}'",
                success=True,
                metadata={
                    "function": "index_document",
                    "chunks_created": len(vector_docs),
                    "collection": collection_name,
                    "source": source,
                    "processing_time_ms": processing_time,
                },
            )

        except Exception as e:
            logger.error(f"Document indexing failed: {e}", exc_info=True)
            return CapabilityResult(content=f"Failed to index document: {str(e)}", success=False, error=str(e))

    async def _handle_semantic_search_function(self, task, context: CapabilityContext) -> CapabilityResult:
        """Handle the semantic_search AI function.

        Args:
            task: Task object
            context: Skill context with parameters

        Returns:
            CapabilityResult with search results
        """
        try:
            start_time = time.time()

            # Extract parameters from task metadata (where function arguments are placed)
            # First try the task metadata (new format), then fall back to context metadata (legacy)
            if hasattr(task, "metadata") and task.metadata:
                params = task.metadata
            else:
                params = context.metadata.get("parameters", {})

            query = params.get("query")
            collection_name = params.get("collection", self.config.default_collection if self.config else "default")
            k = min(params.get("k", 5), self.config.max_k if self.config else 100)
            similarity_threshold = params.get("similarity_threshold", 0.0)
            filters = params.get("filters", {})

            # Validate input
            if not query or len(query.strip()) == 0:
                return CapabilityResult(
                    content="Error: Search query cannot be empty", success=False, error="EMPTY_QUERY"
                )

            # Initialize backends if needed
            if not self.embedding_backend or not self.vector_backend:
                try:
                    # Use context config if available, otherwise use stored config
                    config_to_use = context.config if context.config else (self.config.dict() if self.config else None)
                    if not config_to_use:
                        return CapabilityResult(
                            content="Error: No configuration available for RAG plugin", success=False, error="NO_CONFIG"
                        )
                    await self._initialize_backends(config_to_use)
                except Exception as e:
                    logger.error(f"Failed to initialize RAG backends: {e}")
                    return CapabilityResult(
                        content=f"Error: Failed to initialize RAG backends: {str(e)}",
                        success=False,
                        error="BACKEND_INIT_FAILED",
                    )

            # Generate query embedding
            query_embedding = await self.embedding_backend.embed_text(query)

            # Perform vector search
            search_result = await self.vector_backend.search(
                query_vector=query_embedding, k=k, collection=collection_name, filters=filters if filters else None
            )

            # Filter by similarity threshold
            filtered_docs = []
            filtered_scores = []
            for doc, score in zip(search_result.documents, search_result.scores, strict=False):
                if score >= similarity_threshold:
                    filtered_docs.append(doc)
                    filtered_scores.append(score)

            # Format results
            results = []
            for doc, score in zip(filtered_docs, filtered_scores, strict=False):
                result = {
                    "content": doc.content,
                    "score": score,
                    "source": doc.source,
                    "chunk_index": doc.chunk_index,
                    "metadata": doc.metadata if self.config and self.config.include_metadata else {},
                }
                results.append(result)

            search_time = (time.time() - start_time) * 1000  # milliseconds

            if not results:
                content = f"No documents found matching query: '{query}'"
            else:
                content = f"Found {len(results)} relevant documents for query: '{query}'"

            return CapabilityResult(
                content=content,
                success=True,
                metadata={
                    "function": "semantic_search",
                    "results": results,
                    "query": query,
                    "collection": collection_name,
                    "search_time_ms": search_time,
                    "total_results": len(filtered_docs),
                },
            )

        except Exception as e:
            logger.error(f"Semantic search failed: {e}", exc_info=True)
            return CapabilityResult(content=f"Search failed: {str(e)}", success=False, error=str(e))

    async def _handle_ask_documents_function(self, task, context: CapabilityContext) -> CapabilityResult:
        """Handle the ask_documents AI function.

        Args:
            task: Task object
            context: Skill context with parameters

        Returns:
            CapabilityResult with answer
        """
        try:
            start_time = time.time()

            # Extract parameters from task metadata (where function arguments are placed)
            # First try the task metadata (new format), then fall back to context metadata (legacy)
            if hasattr(task, "metadata") and task.metadata:
                params = task.metadata
            else:
                params = context.metadata.get("parameters", {})

            question = params.get("question")
            collection_name = params.get("collection", self.config.default_collection if self.config else "default")
            max_context_length = params.get(
                "max_context_length", self.config.max_context_length if self.config else 4000
            )
            include_sources = params.get("include_sources", self.config.include_sources if self.config else True)

            # Validate input
            if not question or len(question.strip()) == 0:
                return CapabilityResult(
                    content="Error: Question cannot be empty", success=False, error="EMPTY_QUESTION"
                )

            # Initialize backends if needed
            if not self.embedding_backend or not self.vector_backend:
                try:
                    # Use context config if available, otherwise use stored config
                    config_to_use = context.config if context.config else (self.config.dict() if self.config else None)
                    if not config_to_use:
                        return CapabilityResult(
                            content="Error: No configuration available for RAG plugin", success=False, error="NO_CONFIG"
                        )
                    await self._initialize_backends(config_to_use)
                except Exception as e:
                    logger.error(f"Failed to initialize RAG backends: {e}")
                    return CapabilityResult(
                        content=f"Error: Failed to initialize RAG backends: {str(e)}",
                        success=False,
                        error="BACKEND_INIT_FAILED",
                    )

            # Perform semantic search to get relevant context
            search_context = CapabilityContext(
                task=context.task,
                metadata={
                    "parameters": {
                        "query": question,
                        "collection": collection_name,
                        "k": 10,  # Get more results for context
                        "similarity_threshold": 0.3,
                    }
                },
            )

            search_result = await self._handle_semantic_search_function(task, search_context)
            if not search_result.success:
                return search_result

            # Extract relevant documents
            relevant_docs = search_result.metadata.get("results", [])
            if not relevant_docs:
                return CapabilityResult(
                    content="I couldn't find any relevant information to answer your question. Please try rephrasing your question or check if relevant documents have been indexed.",
                    success=True,
                    metadata={"function": "ask_documents", "question": question, "sources": [], "context_used": 0},
                )

            # Build context for LLM
            context_parts = []
            sources = []
            current_length = 0

            for doc in relevant_docs:
                doc_content = doc["content"]
                doc_length = len(doc_content.split())

                if current_length + doc_length > max_context_length:
                    break

                context_parts.append(f"Source: {doc['source']}\nContent: {doc_content}")
                sources.append(
                    {
                        "source": doc["source"],
                        "score": doc["score"],
                        "chunk_index": doc["chunk_index"],
                        "content_preview": doc_content[:200] + "..." if len(doc_content) > 200 else doc_content,
                    }
                )
                current_length += doc_length

            if not context_parts:
                return CapabilityResult(
                    content="The relevant documents found were too long to process. Please try a more specific question.",
                    success=True,
                    metadata={"function": "ask_documents", "question": question, "sources": [], "context_used": 0},
                )

            context = "\n\n".join(context_parts)

            # Get LLM service for answer generation
            llm_service = self._get_llm_service(context)
            if not llm_service:
                # If no LLM service is available, return a summary based on search results
                answer = self._generate_summary_from_results(relevant_docs, question)
            else:
                # Generate answer using LLM
                answer = await self._generate_llm_answer(llm_service, question, context)

            # Add source citations if requested
            if include_sources and sources:
                citations = "\n\nSources:\n"
                for i, source in enumerate(sources, 1):
                    citations += f"{i}. {source['source']} (similarity: {source['score']:.2f})\n"
                answer += citations

            processing_time = (time.time() - start_time) * 1000

            return CapabilityResult(
                content=answer,
                success=True,
                metadata={
                    "function": "ask_documents",
                    "question": question,
                    "sources": sources if include_sources else [],
                    "context_length": current_length,
                    "processing_time_ms": processing_time,
                },
            )

        except Exception as e:
            logger.error(f"RAG question answering failed: {e}", exc_info=True)
            return CapabilityResult(content=f"Failed to answer question: {str(e)}", success=False, error=str(e))

    def _get_llm_service(self, context: CapabilityContext):
        """Get LLM service from AgentUp services.

        Args:
            context: Skill context

        Returns:
            LLM service instance or None
        """
        return self.services.get("llm_service") or self.services.get("ai_service")

    async def _generate_llm_answer(self, llm_service, question: str, context: str) -> str:
        """Generate answer using LLM service.

        Args:
            llm_service: LLM service instance
            question: User question
            context: Relevant context from documents

        Returns:
            Generated answer
        """
        rag_prompt = f"""Based on the following context, please answer the user's question. If the context doesn't contain enough information to answer the question, please say so clearly.

Context:
{context}

Question: {question}

Please provide a helpful and accurate answer based on the context above."""

        try:
            from agent.llm_providers.base import ChatMessage

            messages = [
                ChatMessage(
                    role="system",
                    content="You are a helpful assistant that answers questions based on provided context.",
                ),
                ChatMessage(role="user", content=rag_prompt),
            ]

            llm_response = await llm_service.chat_complete(messages)
            return llm_response.content

        except Exception as e:
            logger.warning(f"LLM answer generation failed, falling back to summary: {e}")
            return self._generate_summary_from_results([], question)

    def _generate_summary_from_results(self, results: list[dict], question: str) -> str:
        """Generate a summary from search results when LLM is not available.

        Args:
            results: Search results
            question: User question

        Returns:
            Summary text
        """
        if not results:
            return "I couldn't find any relevant information to answer your question."

        summary = (
            f"Based on the documents I found, here are the most relevant excerpts for your question '{question}':\n\n"
        )

        for i, result in enumerate(results[:3], 1):  # Show top 3 results
            content = result["content"]
            source = result["source"]
            score = result["score"]

            # Truncate content if too long
            if len(content) > 300:
                content = content[:297] + "..."

            summary += f"{i}. From {source} (relevance: {score:.2f}):\n{content}\n\n"

        summary += "Please note: This is a summary of relevant document excerpts. For a more comprehensive answer, an LLM service is recommended."

        return summary

    async def _handle_list_collections_function(self, task, context: CapabilityContext) -> CapabilityResult:
        """Handle the list_collections AI function.

        Args:
            task: Task object
            context: Skill context with parameters

        Returns:
            CapabilityResult with collection list
        """
        try:
            if not self.vector_backend:
                return CapabilityResult(
                    content="Vector backend not initialized",
                    success=False,
                    error="Backend not available",
                )

            collections = await self.vector_backend.list_collections()

            if not collections:
                content = "No collections found. You can create collections by indexing documents."
            else:
                content = f"Available collections: {', '.join(collections)}"

            return CapabilityResult(
                content=content,
                success=True,
                metadata={
                    "function": "list_collections",
                    "collections": collections,
                },
            )

        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            return CapabilityResult(
                content=f"Failed to list collections: {str(e)}",
                success=False,
                error=str(e),
            )

    @hookimpl
    def get_state_schema(self) -> dict[str, Any]:
        """Define state schema for the RAG plugin.

        Returns:
            JSON schema for plugin state
        """
        return {
            "type": "object",
            "properties": {
                "last_search_query": {"type": "string", "description": "Last search query performed"},
                "last_search_time": {
                    "type": "string",
                    "format": "date-time",
                    "description": "Timestamp of last search",
                },
                "search_count": {"type": "integer", "description": "Total number of searches performed"},
                "index_count": {"type": "integer", "description": "Total number of documents indexed"},
                "preferred_collection": {"type": "string", "description": "User's preferred collection"},
            },
        }

    @hookimpl
    def get_middleware_config(self) -> list[dict[str, Any]]:
        """Request middleware for the RAG plugin.

        Returns:
            list of middleware configurations
        """
        return [
            {
                "type": "rate_limit",
                "requests_per_minute": 120,  # Higher limit for RAG operations
            },
            {
                "type": "cache",
                "ttl": 600,  # 10 minute cache for search results
            },
            {
                "type": "retry",
                "max_retries": 3,
                "backoff_factor": 2,
            },
            {
                "type": "logging",
                "level": "INFO",
                "include_params": True,
            },
        ]

    async def cleanup(self) -> None:
        """Clean up plugin resources."""
        if self.embedding_backend:
            await self.embedding_backend.close()

        if self.vector_backend:
            await self.vector_backend.close()

        if self.http_client and hasattr(self.http_client, "aclose"):
            await self.http_client.aclose()

        logger.info("RAG plugin cleanup completed")
