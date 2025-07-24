"""
Pinecone vector store backend implementation.

This module provides a production-ready Pinecone integration for the AgentUp RAG plugin,
supporting cloud-hosted vector databases with high availability and advanced querying.
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Any

try:
    import pinecone
    from pinecone import Pinecone, PodSpec, ServerlessSpec

    PINECONE_AVAILABLE = True
except ImportError:
    pinecone = None
    Pinecone = None
    ServerlessSpec = None
    PodSpec = None
    PINECONE_AVAILABLE = False

from ...models import SearchResult, VectorDocument
from ..base import VectorStoreBackend, VectorStoreError

logger = logging.getLogger(__name__)


class PineconeVectorStoreBackend(VectorStoreBackend):
    """Pinecone vector store backend for production-scale deployments.

    Features:
    - Managed cloud infrastructure
    - High availability and auto-scaling
    - Advanced querying and filtering
    - Real-time updates and deletes
    - Namespace support for multi-tenancy
    - Serverless and pod-based options
    """

    def __init__(self, config: dict[str, Any]):
        """Initialize Pinecone vector store backend.

        Args:
            config: Configuration dictionary with Pinecone settings
        """
        super().__init__(config)

        if not PINECONE_AVAILABLE:
            raise VectorStoreError("Pinecone is not available. Install with: pip install pinecone-client")

        self.client = None
        self.index = None
        self._initialized = False

        # Configuration
        self.api_key = config.get("api_key")
        self.environment = config.get("environment")
        self.index_name = config.get("index_name")
        self.dimension = config.get("dimension")
        self.metric = config.get("metric", "cosine")
        self.pod_type = config.get("pod_type", "p1.x1")
        self.replicas = config.get("replicas", 1)
        self.shards = config.get("shards", 1)
        self.pods = config.get("pods", 1)
        self.metadata_config = config.get("metadata_config", {})
        self.source_tag = config.get("source_tag")

        # Namespace support (for multi-tenancy)
        self.default_namespace = config.get("default_namespace", "")

        # Serverless vs Pod configuration
        self.deployment_type = config.get("deployment_type", "serverless")  # "serverless" or "pod"
        self.cloud = config.get("cloud", "aws")
        self.region = config.get("region", "us-east-1")

        logger.info(f"Initialized Pinecone backend with config: {self._get_safe_config()}")

    def _validate_config(self) -> None:
        """Validate Pinecone configuration."""
        required_fields = ["api_key", "index_name", "dimension"]
        for field in required_fields:
            value = getattr(self, field)
            if not value:
                raise ValueError(f"Pinecone {field} is required")

        # Validate dimension
        if not isinstance(self.dimension, int) or self.dimension <= 0:
            raise ValueError("Dimension must be a positive integer")

        # Validate metric
        valid_metrics = ["cosine", "euclidean", "dotproduct"]
        if self.metric not in valid_metrics:
            raise ValueError(f"Metric must be one of: {valid_metrics}")

        # Validate deployment type
        if self.deployment_type not in ["serverless", "pod"]:
            raise ValueError("Deployment type must be 'serverless' or 'pod'")

        # For pod deployment, validate pod configuration
        if self.deployment_type == "pod":
            if not self.environment:
                raise ValueError("Environment is required for pod deployment")

            if not isinstance(self.replicas, int) or self.replicas < 1:
                raise ValueError("Replicas must be a positive integer")

    def _get_safe_config(self) -> dict[str, Any]:
        """Get configuration with sensitive data masked."""
        safe_config = self.config.copy()
        if "api_key" in safe_config and safe_config["api_key"]:
            safe_config["api_key"] = "***"
        return safe_config

    async def initialize(self) -> None:
        """Initialize Pinecone client and index."""
        if self._initialized:
            return

        try:
            # Initialize Pinecone client
            self.client = Pinecone(api_key=self.api_key)

            # Check if index exists, create if not
            await self._ensure_index_exists()

            # Connect to index
            self.index = self.client.Index(self.index_name)

            # Test connection
            await self._run_async(self.index.describe_index_stats)

            self._initialized = True
            logger.info(f"Pinecone backend initialized successfully with index '{self.index_name}'")

        except Exception as e:
            logger.error(f"Failed to initialize Pinecone backend: {e}")
            raise VectorStoreError(f"Pinecone initialization failed: {e}") from e

    async def _run_async(self, func, *args, **kwargs):
        """Run Pinecone operations asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, func, *args, **kwargs)

    async def _ensure_index_exists(self) -> None:
        """Ensure the Pinecone index exists, create if not."""
        try:
            # list existing indexes
            indexes = await self._run_async(self.client.list_indexes)
            index_names = [idx.name for idx in indexes.indexes]

            if self.index_name in index_names:
                logger.info(f"Pinecone index '{self.index_name}' already exists")
                return

            # Create index specification based on deployment type
            if self.deployment_type == "serverless":
                spec = ServerlessSpec(cloud=self.cloud, region=self.region)
                logger.info(f"Creating serverless index '{self.index_name}' in {self.cloud}/{self.region}")
            else:
                # Pod-based deployment
                spec = PodSpec(
                    environment=self.environment,
                    pod_type=self.pod_type,
                    pods=self.pods,
                    replicas=self.replicas,
                    shards=self.shards,
                    metadata_config=self.metadata_config,
                    source_tag=self.source_tag,
                )
                logger.info(f"Creating pod-based index '{self.index_name}' in {self.environment}")

            # Create the index
            await self._run_async(
                self.client.create_index, name=self.index_name, dimension=self.dimension, metric=self.metric, spec=spec
            )

            # Wait for index to be ready
            await self._wait_for_index_ready()

            logger.info(f"Successfully created Pinecone index '{self.index_name}'")

        except Exception as e:
            logger.error(f"Failed to ensure index exists: {e}")
            raise VectorStoreError(f"Index creation failed: {e}") from e

    async def _wait_for_index_ready(self, timeout: int = 300) -> None:
        """Wait for index to be ready for operations.

        Args:
            timeout: Maximum time to wait in seconds
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                # Check index status
                description = await self._run_async(self.client.describe_index, self.index_name)

                if hasattr(description, "status") and description.status.ready:
                    logger.info(f"Index '{self.index_name}' is ready")
                    return

                logger.debug(f"Index '{self.index_name}' not ready yet, waiting...")
                await asyncio.sleep(5)

            except Exception as e:
                logger.warning(f"Error checking index status: {e}")
                await asyncio.sleep(5)

        raise VectorStoreError(f"Index '{self.index_name}' did not become ready within {timeout} seconds")

    async def add_vectors(self, vectors: list[VectorDocument], collection: str | None = None) -> None:
        """Add vector documents to Pinecone index.

        Args:
            vectors: list of vector documents to add
            collection: Collection name (used as namespace)
        """
        if not self._initialized:
            await self.initialize()

        if not vectors:
            return

        namespace = collection or self.default_namespace

        try:
            # Prepare vectors for upsert
            upsert_data = []

            for doc in vectors:
                # Prepare metadata
                metadata = {
                    "content": doc.content,
                    "source": doc.source,
                    "chunk_index": doc.chunk_index,
                    "created_at": doc.created_at.isoformat(),
                    **doc.metadata,
                }

                # Pinecone has metadata size limits, so we may need to truncate content
                if len(metadata.get("content", "")) > 40000:  # Pinecone limit
                    metadata["content"] = metadata["content"][:40000] + "..."
                    metadata["content_truncated"] = True

                # Ensure all metadata values are serializable
                metadata = self._sanitize_metadata(metadata)

                upsert_data.append({"id": doc.id, "values": doc.embedding, "metadata": metadata})

            # Upsert in batches to avoid size limits
            batch_size = 100  # Pinecone recommended batch size

            for i in range(0, len(upsert_data), batch_size):
                batch = upsert_data[i : i + batch_size]

                await self._run_async(self.index.upsert, vectors=batch, namespace=namespace)

                logger.debug(f"Upserted batch {i // batch_size + 1} ({len(batch)} vectors) to namespace '{namespace}'")

            logger.info(f"Added {len(vectors)} vectors to Pinecone namespace '{namespace}'")

        except Exception as e:
            logger.error(f"Failed to add vectors to Pinecone: {e}")
            raise VectorStoreError(f"Failed to add vectors: {e}") from e

    def _sanitize_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """Sanitize metadata for Pinecone compatibility.

        Pinecone supports strings, numbers, booleans, and lists of these types.
        """
        sanitized = {}

        for key, value in metadata.items():
            if isinstance(value, str | int | float | bool):
                sanitized[key] = value
            elif isinstance(value, list):
                # Filter list to only supported types
                filtered_list = [item for item in value if isinstance(item, str | int | float | bool)]
                if filtered_list:
                    sanitized[key] = filtered_list
            elif value is None:
                sanitized[key] = ""
            else:
                # Convert to string
                sanitized[key] = str(value)

        return sanitized

    async def search(
        self,
        query_vector: list[float],
        k: int = 5,
        collection: str | None = None,
        filters: dict[str, Any] | None = None,
    ) -> SearchResult:
        """Search for similar vectors in Pinecone.

        Args:
            query_vector: Vector to search for
            k: Number of results to return
            collection: Collection to search in (used as namespace)
            filters: Metadata filters to apply

        Returns:
            SearchResult with matching documents and scores
        """
        if not self._initialized:
            await self.initialize()

        namespace = collection or self.default_namespace
        start_time = time.time()

        try:
            # Build filter for Pinecone
            pinecone_filter = None
            if filters:
                pinecone_filter = self._build_pinecone_filter(filters)

            # Perform query
            query_result = await self._run_async(
                self.index.query,
                vector=query_vector,
                top_k=k,
                namespace=namespace,
                filter=pinecone_filter,
                include_metadata=True,
                include_values=True,
            )

            # Convert results to VectorDocument objects
            documents = []
            scores = []

            for match in query_result.matches:
                metadata = match.metadata or {}

                # Extract required fields from metadata
                content = metadata.pop("content", "")
                source = metadata.pop("source", "unknown")
                chunk_index = metadata.pop("chunk_index", 0)
                created_at_str = metadata.pop("created_at", None)

                # Parse created_at
                created_at = None
                if created_at_str:
                    try:
                        created_at = datetime.fromisoformat(created_at_str)
                    except (ValueError, TypeError):
                        created_at = None

                if not created_at:
                    created_at = datetime.utcnow()

                # Create VectorDocument
                vector_doc = VectorDocument(
                    id=match.id,
                    content=content,
                    embedding=match.values if match.values else [],
                    metadata=metadata,
                    source=source,
                    chunk_index=int(chunk_index) if isinstance(chunk_index, int | str) else 0,
                    created_at=created_at,
                )

                documents.append(vector_doc)
                scores.append(float(match.score))

            search_time = (time.time() - start_time) * 1000

            logger.debug(f"Pinecone search returned {len(documents)} results in {search_time:.2f}ms")

            return SearchResult(
                documents=documents,
                scores=scores,
                query="",  # Query vector doesn't have text representation
                search_time=search_time,
                total_results=len(documents),
                collection=collection,
                filters=filters,
            )

        except Exception as e:
            logger.error(f"Pinecone search failed: {e}")
            raise VectorStoreError(f"Search failed: {e}") from e

    def _build_pinecone_filter(self, filters: dict[str, Any]) -> dict[str, Any]:
        """Build Pinecone filter from standard filter dictionary.

        Args:
            filters: Standard filter dictionary

        Returns:
            Pinecone-compatible filter
        """
        pinecone_filter = {}

        for key, value in filters.items():
            if isinstance(value, dict):
                # Handle operators
                if "$eq" in value:
                    pinecone_filter[key] = {"$eq": value["$eq"]}
                elif "$ne" in value:
                    pinecone_filter[key] = {"$ne": value["$ne"]}
                elif "$gt" in value:
                    pinecone_filter[key] = {"$gt": value["$gt"]}
                elif "$gte" in value:
                    pinecone_filter[key] = {"$gte": value["$gte"]}
                elif "$lt" in value:
                    pinecone_filter[key] = {"$lt": value["$lt"]}
                elif "$lte" in value:
                    pinecone_filter[key] = {"$lte": value["$lte"]}
                elif "$in" in value:
                    pinecone_filter[key] = {"$in": value["$in"]}
                elif "$nin" in value:
                    pinecone_filter[key] = {"$nin": value["$nin"]}
                else:
                    # Direct assignment for other dict values
                    pinecone_filter[key] = value
            elif isinstance(value, list):
                # Convert list to $in operator
                pinecone_filter[key] = {"$in": value}
            else:
                # Direct equality
                pinecone_filter[key] = {"$eq": value}

        return pinecone_filter

    async def delete_vectors(self, vector_ids: list[str], collection: str | None = None) -> None:
        """Delete vectors by their IDs.

        Args:
            vector_ids: list of vector IDs to delete
            collection: Collection to delete from (used as namespace)
        """
        if not self._initialized:
            await self.initialize()

        if not vector_ids:
            return

        namespace = collection or self.default_namespace

        try:
            # Delete vectors in batches
            batch_size = 1000  # Pinecone limit for delete operations

            for i in range(0, len(vector_ids), batch_size):
                batch = vector_ids[i : i + batch_size]

                await self._run_async(self.index.delete, ids=batch, namespace=namespace)

                logger.debug(f"Deleted batch {i // batch_size + 1} ({len(batch)} vectors) from namespace '{namespace}'")

            logger.info(f"Deleted {len(vector_ids)} vectors from Pinecone namespace '{namespace}'")

        except Exception as e:
            logger.error(f"Failed to delete vectors from Pinecone: {e}")
            raise VectorStoreError(f"Failed to delete vectors: {e}") from e

    async def get_collection_stats(self, collection: str | None = None) -> dict[str, Any]:
        """Get statistics about a collection (namespace).

        Args:
            collection: Collection name (namespace)

        Returns:
            dictionary containing collection statistics
        """
        if not self._initialized:
            await self.initialize()

        namespace = collection or self.default_namespace

        try:
            # Get index stats
            stats = await self._run_async(self.index.describe_index_stats)

            # Get namespace-specific stats
            namespace_stats = None
            if hasattr(stats, "namespaces") and namespace in stats.namespaces:
                namespace_stats = stats.namespaces[namespace]
            elif not namespace and hasattr(stats, "total_vector_count"):
                # Default namespace stats
                namespace_stats = {"vector_count": stats.total_vector_count}

            return {
                "name": namespace or "default",
                "exists": namespace_stats is not None,
                "count": namespace_stats.vector_count if namespace_stats else 0,
                "dimension": self.dimension,
                "metric": self.metric,
                "index_fullness": getattr(stats, "index_fullness", 0.0),
                "total_vector_count": getattr(stats, "total_vector_count", 0),
            }

        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            raise VectorStoreError(f"Failed to get collection stats: {e}") from e

    async def list_collections(self) -> list[str]:
        """list all available collections (namespaces).

        Returns:
            list of collection names (namespaces)
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Get index stats to see available namespaces
            stats = await self._run_async(self.index.describe_index_stats)

            namespaces = []
            if hasattr(stats, "namespaces"):
                namespaces = list(stats.namespaces.keys())

            # Add default namespace if it has vectors but isn't explicitly listed
            if hasattr(stats, "total_vector_count") and stats.total_vector_count > 0 and "" not in namespaces:
                namespaces.append("")

            # Filter out empty string for cleaner output
            collections = [ns for ns in namespaces if ns] or ["default"]

            logger.debug(f"Found {len(collections)} collections: {collections}")
            return collections

        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            raise VectorStoreError(f"Failed to list collections: {e}") from e

    async def create_collection(self, name: str, dimension: int, description: str | None = None) -> None:
        """Create a new collection (namespace).

        Note: In Pinecone, namespaces are created implicitly when vectors are added.
        This method validates the parameters but doesn't create anything.

        Args:
            name: Collection name (namespace)
            dimension: Vector dimension (must match index dimension)
            description: Optional description (stored as metadata)
        """
        if not self._initialized:
            await self.initialize()

        # Validate dimension matches index
        if dimension != self.dimension:
            raise VectorStoreError(f"Collection dimension {dimension} must match index dimension {self.dimension}")

        # In Pinecone, namespaces are created implicitly
        # We'll just validate the name and log
        if not name or not isinstance(name, str):
            raise ValueError("Collection name must be a non-empty string")

        logger.info(f"Collection '{name}' will be created when first vector is added")

    async def delete_collection(self, name: str) -> None:
        """Delete a collection (namespace) and all its vectors.

        Args:
            name: Collection name (namespace) to delete
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Delete all vectors in the namespace
            await self._run_async(self.index.delete, delete_all=True, namespace=name)

            logger.info(f"Deleted all vectors from namespace '{name}'")

        except Exception as e:
            logger.error(f"Failed to delete collection '{name}': {e}")
            raise VectorStoreError(f"Failed to delete collection: {e}") from e

    async def close(self) -> None:
        """Clean up Pinecone resources."""
        try:
            # Pinecone client doesn't require explicit cleanup
            self.index = None
            self.client = None
            self._initialized = False

            logger.info("Pinecone backend closed successfully")

        except Exception as e:
            logger.warning(f"Error closing Pinecone backend: {e}")
