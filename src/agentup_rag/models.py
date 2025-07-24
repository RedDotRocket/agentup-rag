"""
Data models for the AgentUp RAG plugin.

This module defines Pydantic models for all data structures used in the RAG system,
including vector documents, search results, collections, and configuration schemas.
"""

import hashlib
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class ChunkingStrategy(str, Enum):
    """Available text chunking strategies."""

    FIXED = "fixed"
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"


class SimilarityMetric(str, Enum):
    """Available similarity metrics for vector search."""

    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot_product"


class VectorDocument(BaseModel):
    """Represents a document chunk with its vector embedding."""

    id: str = Field(..., description="Unique identifier for the document")
    content: str = Field(..., description="Original text content")
    embedding: list[float] = Field(..., description="Vector embedding of the content")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    source: str = Field(..., description="Original document source")
    chunk_index: int = Field(..., description="Position of this chunk in the original document")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime | None = Field(None, description="Last update timestamp")

    @field_validator("id")
    @classmethod
    def validate_id(cls, v):
        """Ensure ID is not empty."""
        if not v or not v.strip():
            raise ValueError("Document ID cannot be empty")
        return v.strip()

    @field_validator("content")
    @classmethod
    def validate_content(cls, v):
        """Ensure content is not empty."""
        if not v or not v.strip():
            raise ValueError("Document content cannot be empty")
        return v.strip()

    @field_validator("embedding")
    @classmethod
    def validate_embedding(cls, v):
        """Ensure embedding is not empty and contains valid numbers."""
        if not v:
            raise ValueError("Embedding cannot be empty")
        if not all(isinstance(x, int | float) for x in v):
            raise ValueError("Embedding must contain only numeric values")
        return v

    @classmethod
    def generate_id(cls, content: str, source: str, chunk_index: int) -> str:
        """Generate a deterministic ID for a document chunk.

        Args:
            content: The document content
            source: The document source
            chunk_index: The chunk index

        Returns:
            A unique document ID

        Bandit: B324
        Not used in with secure context.
        """
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]  # nosec
        return f"{source}_{chunk_index}_{content_hash}"

    class Config:
        json_encoders = {datetime: lambda dt: dt.isoformat()}


class SearchResult(BaseModel):
    """Result from vector similarity search."""

    documents: list[VectorDocument] = Field(..., description="Matching documents")
    scores: list[float] = Field(..., description="Similarity scores for each document")
    query: str = Field(..., description="Original search query")
    search_time: float = Field(..., description="Search time in milliseconds")
    total_results: int = Field(..., description="Total number of matches before limiting")
    collection: str | None = Field(None, description="Collection that was searched")
    filters: dict[str, Any] | None = Field(None, description="Filters that were applied")

    @field_validator("scores")
    @classmethod
    def validate_documents_scores_match(cls, v, info):
        """Ensure documents and scores have the same length."""
        # Note: Cross-field validation will be done in model validator
        return v

    @field_validator("scores", mode="after")
    @classmethod
    def validate_scores_range(cls, v):
        """Ensure all scores are valid numbers."""
        if not all(isinstance(x, int | float) and 0 <= x <= 1 for x in v):
            raise ValueError("All scores must be numbers between 0 and 1")
        return v

    @property
    def best_score(self) -> float | None:
        """Get the highest similarity score."""
        return max(self.scores) if self.scores else None

    @property
    def average_score(self) -> float | None:
        """Get the average similarity score."""
        return sum(self.scores) / len(self.scores) if self.scores else None

    @model_validator(mode="after")
    def validate_documents_scores_length(self):
        """Ensure documents and scores have the same length."""
        if len(self.documents) != len(self.scores):
            raise ValueError("Number of documents and scores must match")
        return self


class Collection(BaseModel):
    """Represents a collection of documents."""

    name: str = Field(..., description="Collection name")
    description: str = Field("", description="Collection description")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    document_count: int = Field(0, description="Number of documents in collection")
    total_tokens: int = Field(0, description="Total number of tokens across all documents")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Collection metadata")
    embedding_dimension: int | None = Field(None, description="Vector dimension for this collection")

    @field_validator("name")
    @classmethod
    def validate_name(cls, v):
        """Ensure collection name is valid."""
        if not v or not v.strip():
            raise ValueError("Collection name cannot be empty")
        # Allow alphanumeric, hyphens, underscores
        import re

        if not re.match(r"^[a-zA-Z0-9_-]+$", v.strip()):
            raise ValueError("Collection name can only contain letters, numbers, hyphens, and underscores")
        return v.strip()

    @field_validator("document_count", "total_tokens")
    @classmethod
    def validate_non_negative(cls, v):
        """Ensure counts are non-negative."""
        if v < 0:
            raise ValueError("Counts cannot be negative")
        return v

    class Config:
        json_encoders = {datetime: lambda dt: dt.isoformat()}


class Chunk(BaseModel):
    """Represents a text chunk from document processing."""

    content: str = Field(..., description="Chunk text content")
    chunk_index: int = Field(..., description="Index of this chunk in the document")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Chunk metadata")
    start_char: int | None = Field(None, description="Starting character position in original document")
    end_char: int | None = Field(None, description="Ending character position in original document")

    @field_validator("content")
    @classmethod
    def validate_content(cls, v):
        """Ensure chunk content is not empty."""
        if not v or not v.strip():
            raise ValueError("Chunk content cannot be empty")
        return v.strip()

    @field_validator("chunk_index")
    @classmethod
    def validate_chunk_index(cls, v):
        """Ensure chunk index is non-negative."""
        if v < 0:
            raise ValueError("Chunk index cannot be negative")
        return v


class DocumentProcessingResult(BaseModel):
    """Result from document processing operations."""

    success: bool = Field(..., description="Whether processing was successful")
    chunks_created: int = Field(0, description="Number of chunks created")
    source: str = Field(..., description="Source document identifier")
    processing_time: float = Field(..., description="Processing time in milliseconds")
    error: str | None = Field(None, description="Error message if processing failed")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional processing metadata")


class EmbeddingConfig(BaseModel):
    """Base configuration for embedding backends."""

    model: str = Field(..., description="Model identifier")
    batch_size: int = Field(32, description="Batch size for embedding generation")
    rate_limit: int | None = Field(None, description="Rate limit (requests per minute)")
    timeout: int = Field(30, description="Request timeout in seconds")

    @field_validator("batch_size")
    @classmethod
    def validate_batch_size(cls, v):
        """Ensure batch size is positive."""
        if v <= 0:
            raise ValueError("Batch size must be positive")
        return v

    @field_validator("timeout")
    @classmethod
    def validate_timeout(cls, v):
        """Ensure timeout is positive."""
        if v <= 0:
            raise ValueError("Timeout must be positive")
        return v


class OpenAIEmbeddingConfig(EmbeddingConfig):
    """Configuration for OpenAI embedding backend."""

    api_key: str = Field(..., description="OpenAI API key")
    organization: str | None = Field(None, description="OpenAI organization ID")
    max_retries: int = Field(3, description="Maximum number of retries")
    backoff_factor: float = Field(2.0, description="Exponential backoff factor")

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, v):
        """Ensure API key is not empty."""
        if not v or not v.strip():
            raise ValueError("OpenAI API key cannot be empty")
        return v.strip()


class LocalEmbeddingConfig(EmbeddingConfig):
    """Configuration for local embedding backend."""

    device: str = Field("cpu", description="Device to run model on (cpu, cuda, mps)")
    cache_dir: str | None = Field(None, description="Directory to cache models")
    normalize_embeddings: bool = Field(True, description="Whether to normalize embeddings")

    @field_validator("device")
    @classmethod
    def validate_device(cls, v):
        """Ensure device is valid."""
        valid_devices = ["cpu", "cuda", "mps"]
        if v not in valid_devices:
            raise ValueError(f"Device must be one of: {valid_devices}")
        return v


class VectorStoreConfig(BaseModel):
    """Base configuration for vector store backends."""

    similarity_metric: SimilarityMetric = Field(SimilarityMetric.COSINE, description="Similarity metric to use")
    timeout: int = Field(30, description="Request timeout in seconds")

    @field_validator("timeout")
    @classmethod
    def validate_timeout(cls, v):
        """Ensure timeout is positive."""
        if v <= 0:
            raise ValueError("Timeout must be positive")
        return v


class MemoryVectorStoreConfig(VectorStoreConfig):
    """Configuration for in-memory vector store."""

    index_type: str = Field("flat", description="FAISS index type (flat, ivf, hnsw)")
    persist_path: str | None = Field(None, description="Path to persist index")
    auto_save: bool = Field(True, description="Automatically save index changes")
    save_interval: int = Field(300, description="Auto-save interval in seconds")

    @field_validator("index_type")
    @classmethod
    def validate_index_type(cls, v):
        """Ensure index type is valid."""
        valid_types = ["flat", "ivf", "hnsw"]
        if v not in valid_types:
            raise ValueError(f"Index type must be one of: {valid_types}")
        return v


class ChromaVectorStoreConfig(VectorStoreConfig):
    """Configuration for Chroma vector store."""

    persist_directory: str | None = Field(None, description="Directory to persist Chroma data")
    collection_name: str = Field("documents", description="Default collection name")
    host: str | None = Field(None, description="Chroma server host (for remote)")
    port: int | None = Field(None, description="Chroma server port (for remote)")
    ssl: bool = Field(False, description="Use SSL for remote connections")
    api_key: str | None = Field(None, description="API key for authentication")

    @field_validator("collection_name")
    @classmethod
    def validate_collection_name(cls, v):
        """Ensure collection name is valid."""
        if not v or not v.strip():
            raise ValueError("Collection name cannot be empty")
        return v.strip()


class PineconeVectorStoreConfig(VectorStoreConfig):
    """Configuration for Pinecone vector store."""

    api_key: str = Field(..., description="Pinecone API key")
    environment: str = Field(..., description="Pinecone environment")
    index_name: str = Field(..., description="Pinecone index name")
    dimension: int = Field(..., description="Vector dimension")
    metric: str = Field("cosine", description="Distance metric")
    pod_type: str = Field("p1.x1", description="Pod type for index")
    replicas: int = Field(1, description="Number of replicas")

    @field_validator("api_key", "environment", "index_name")
    @classmethod
    def validate_required_fields(cls, v):
        """Ensure required fields are not empty."""
        if not v or not v.strip():
            raise ValueError("Required field cannot be empty")
        return v.strip()

    @field_validator("dimension")
    @classmethod
    def validate_dimension(cls, v):
        """Ensure dimension is positive."""
        if v <= 0:
            raise ValueError("Dimension must be positive")
        return v


class WeaviateVectorStoreConfig(VectorStoreConfig):
    """Configuration for Weaviate vector store."""

    url: str = Field(..., description="Weaviate instance URL")
    api_key: str | None = Field(None, description="Weaviate API key")
    class_name: str = Field("Document", description="Weaviate class name")
    properties: list[dict[str, Any]] = Field(default_factory=list, description="Weaviate class properties")

    @field_validator("url")
    @classmethod
    def validate_url(cls, v):
        """Ensure URL is not empty."""
        if not v or not v.strip():
            raise ValueError("Weaviate URL cannot be empty")
        return v.strip()

    @field_validator("class_name")
    @classmethod
    def validate_class_name(cls, v):
        """Ensure class name is valid."""
        if not v or not v.strip():
            raise ValueError("Class name cannot be empty")
        return v.strip()


class ChunkingConfig(BaseModel):
    """Configuration for text chunking."""

    strategy: ChunkingStrategy = Field(ChunkingStrategy.RECURSIVE, description="Chunking strategy")
    chunk_size: int = Field(1000, description="Target chunk size in characters")
    chunk_overlap: int = Field(200, description="Overlap between chunks in characters")
    separators: list[str] = Field(
        default_factory=lambda: ["\n\n", "\n", ".", "!", "?"], description="Text separators for recursive chunking"
    )
    similarity_threshold: float = Field(0.8, description="Similarity threshold for semantic chunking")

    @field_validator("chunk_size", "chunk_overlap")
    @classmethod
    def validate_positive(cls, v):
        """Ensure values are positive."""
        if v <= 0:
            raise ValueError("Chunk size and overlap must be positive")
        return v

    @field_validator("chunk_overlap")
    @classmethod
    def validate_overlap(cls, v, info):
        """Ensure overlap is less than chunk size."""
        # Note: Cross-field validation will be done in model validator
        return v

    @field_validator("similarity_threshold")
    @classmethod
    def validate_similarity_threshold(cls, v):
        """Ensure similarity threshold is between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError("Similarity threshold must be between 0 and 1")
        return v

    @model_validator(mode="after")
    def validate_chunk_overlap_size(self):
        """Ensure overlap is less than chunk size."""
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("Chunk overlap must be less than chunk size")
        return self


class RAGConfig(BaseModel):
    """Main configuration for the RAG plugin."""

    # Backend configurations
    embedding_backend: str = Field(..., description="Embedding backend type")
    embedding_config: OpenAIEmbeddingConfig | LocalEmbeddingConfig = Field(
        ..., description="Embedding backend configuration"
    )

    vector_backend: str = Field(..., description="Vector store backend type")
    vector_config: (
        MemoryVectorStoreConfig | ChromaVectorStoreConfig | PineconeVectorStoreConfig | WeaviateVectorStoreConfig
    ) = Field(..., description="Vector store backend configuration")

    # Processing configuration
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig, description="Text chunking configuration")

    # Search configuration
    default_k: int = Field(5, description="Default number of search results")
    max_k: int = Field(100, description="Maximum number of search results")
    similarity_threshold: float = Field(0.0, description="Minimum similarity score")
    include_metadata: bool = Field(True, description="Include metadata in search results")

    # RAG configuration
    max_context_length: int = Field(4000, description="Maximum context length for RAG")
    context_overlap: int = Field(100, description="Overlap between context chunks")
    include_sources: bool = Field(True, description="Include source citations")

    # Collection management
    default_collection: str = Field("documents", description="Default collection name")
    auto_create_collections: bool = Field(True, description="Automatically create collections")
    max_collections: int = Field(10, description="Maximum number of collections")

    # Performance settings
    batch_size: int = Field(32, description="Batch size for processing")
    max_concurrent: int = Field(5, description="Maximum concurrent operations")
    timeout: int = Field(30, description="Operation timeout in seconds")

    # Monitoring and logging
    log_queries: bool = Field(True, description="Log search queries")
    log_indexing: bool = Field(True, description="Log document indexing")
    track_performance: bool = Field(True, description="Track performance metrics")
    export_metrics: bool = Field(False, description="Export metrics to monitoring system")

    @field_validator("embedding_backend")
    @classmethod
    def validate_embedding_backend(cls, v):
        """Ensure embedding backend is supported."""
        supported_backends = ["openai", "local"]
        if v not in supported_backends:
            raise ValueError(f"Embedding backend must be one of: {supported_backends}")
        return v

    @field_validator("vector_backend")
    @classmethod
    def validate_vector_backend(cls, v):
        """Ensure vector backend is supported."""
        supported_backends = ["memory", "chroma", "pinecone", "weaviate"]
        if v not in supported_backends:
            raise ValueError(f"Vector backend must be one of: {supported_backends}")
        return v

    @field_validator("default_k", "max_k")
    @classmethod
    def validate_k_values(cls, v):
        """Ensure k values are positive."""
        if v <= 0:
            raise ValueError("k values must be positive")
        return v

    @field_validator("max_k")
    @classmethod
    def validate_max_k(cls, v, info):
        """Ensure max_k is greater than default_k."""
        # Note: Cross-field validation will be done in model validator
        return v

    @model_validator(mode="after")
    def validate_max_k_greater_than_default(self):
        """Ensure max_k is greater than or equal to default_k."""
        if self.max_k < self.default_k:
            raise ValueError("max_k must be greater than or equal to default_k")
        return self
