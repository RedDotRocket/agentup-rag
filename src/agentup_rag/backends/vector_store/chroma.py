"""
Chroma vector store backend implementation.

This module provides a production-ready Chroma integration for the AgentUp RAG plugin,
supporting both local and remote Chroma deployments with full collection management.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

try:
    import chromadb
    from chromadb.config import Settings
    from chromadb.api.models.Collection import Collection as ChromaCollection
    CHROMA_AVAILABLE = True
except ImportError:
    chromadb = None
    Settings = None
    ChromaCollection = None
    CHROMA_AVAILABLE = False

from ..base import VectorStoreBackend, VectorStoreError
from ...models import SearchResult, VectorDocument

logger = logging.getLogger(__name__)


class ChromaVectorStoreBackend(VectorStoreBackend):
    """Chroma vector store backend for local and remote deployments.
    
    Features:
    - SQLite-based persistence for local deployments
    - HTTP API support for remote Chroma servers
    - Collections and metadata support
    - Automatic embedding function integration
    - Advanced querying and filtering
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Chroma vector store backend.
        
        Args:
            config: Configuration dictionary with Chroma settings
        """
        super().__init__(config)
        
        if not CHROMA_AVAILABLE:
            raise VectorStoreError(
                "Chroma is not available. Install with: pip install chromadb"
            )
        
        self.client = None
        self.collections = {}
        self._initialized = False
        
        # Configuration
        self.persist_directory = config.get("persist_directory")
        self.collection_name = config.get("collection_name", "documents")
        self.host = config.get("host")
        self.port = config.get("port", 8000)
        self.ssl = config.get("ssl", False)
        self.api_key = config.get("api_key")
        self.distance_function = config.get("distance_function", "cosine")
        
        logger.info(f"Initialized Chroma backend with config: {self._get_safe_config()}")
    
    def _validate_config(self) -> None:
        """Validate Chroma configuration."""
        # Check if it's remote or local setup
        if self.host:
            # Remote setup
            if not isinstance(self.port, int) or self.port <= 0:
                raise ValueError("Port must be a positive integer for remote Chroma")
            if self.ssl and not self.host.startswith(('https://', 'http://')):
                logger.warning("SSL enabled but host doesn't specify protocol")
        else:
            # Local setup - persist_directory is optional
            pass
        
        # Validate distance function
        valid_distance_functions = ["cosine", "l2", "ip"]
        if self.distance_function not in valid_distance_functions:
            raise ValueError(f"Distance function must be one of: {valid_distance_functions}")
    
    def _get_safe_config(self) -> Dict[str, Any]:
        """Get configuration with sensitive data masked."""
        safe_config = self.config.copy()
        if "api_key" in safe_config and safe_config["api_key"]:
            safe_config["api_key"] = "***"
        return safe_config
    
    async def initialize(self) -> None:
        """Initialize Chroma client and connection."""
        if self._initialized:
            return
        
        try:
            if self.host:
                # Remote Chroma setup
                protocol = "https" if self.ssl else "http"
                url = f"{protocol}://{self.host}:{self.port}"
                
                settings = Settings(
                    chroma_api_impl="rest",
                    chroma_server_host=self.host,
                    chroma_server_http_port=str(self.port),
                    chroma_server_ssl_enabled=self.ssl,
                )
                
                # Add authentication if API key provided
                if self.api_key:
                    settings.chroma_server_auth_credentials = self.api_key
                
                self.client = chromadb.HttpClient(
                    host=self.host,
                    port=self.port,
                    ssl=self.ssl,
                    headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else None,
                    settings=settings
                )
                
                logger.info(f"Connected to remote Chroma at {url}")
            else:
                # Local Chroma setup
                if self.persist_directory:
                    settings = Settings(
                        persist_directory=self.persist_directory,
                        is_persistent=True
                    )
                    self.client = chromadb.PersistentClient(
                        path=self.persist_directory,
                        settings=settings
                    )
                    logger.info(f"Connected to persistent Chroma at {self.persist_directory}")
                else:
                    self.client = chromadb.EphemeralClient()
                    logger.info("Connected to ephemeral Chroma client")
            
            # Test connection by listing collections
            await self._run_sync(self.client.list_collections)
            
            self._initialized = True
            logger.info("Chroma backend initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Chroma backend: {e}")
            raise VectorStoreError(f"Chroma initialization failed: {e}") from e
    
    async def _run_sync(self, func, *args, **kwargs):
        """Run synchronous Chroma operations in thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, func, *args, **kwargs)
    
    async def add_vectors(
        self, 
        vectors: List[VectorDocument],
        collection: Optional[str] = None
    ) -> None:
        """Add vector documents to Chroma collection.
        
        Args:
            vectors: List of vector documents to add
            collection: Collection name (uses default if None)
        """
        if not self._initialized:
            await self.initialize()
        
        if not vectors:
            return
        
        collection_name = collection or self.collection_name
        
        try:
            # Get or create collection
            chroma_collection = await self._get_or_create_collection(
                collection_name, 
                len(vectors[0].embedding)
            )
            
            # Prepare data for Chroma
            ids = [doc.id for doc in vectors]
            embeddings = [doc.embedding for doc in vectors]
            documents = [doc.content for doc in vectors]
            
            # Prepare metadata
            metadatas = []
            for doc in vectors:
                metadata = {
                    "source": doc.source,
                    "chunk_index": doc.chunk_index,
                    "created_at": doc.created_at.isoformat(),
                    **doc.metadata
                }
                # Chroma requires all metadata values to be strings, numbers, or booleans
                metadata = self._sanitize_metadata(metadata)
                metadatas.append(metadata)
            
            # Add to Chroma
            await self._run_sync(
                chroma_collection.add,
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            
            logger.info(f"Added {len(vectors)} vectors to collection '{collection_name}'")
            
        except Exception as e:
            logger.error(f"Failed to add vectors to Chroma: {e}")
            raise VectorStoreError(f"Failed to add vectors: {e}") from e
    
    def _sanitize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize metadata for Chroma compatibility.
        
        Chroma only supports string, int, float, and bool values in metadata.
        """
        sanitized = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                sanitized[key] = value
            elif value is None:
                sanitized[key] = ""
            else:
                # Convert complex types to strings
                sanitized[key] = str(value)
        return sanitized
    
    async def search(
        self,
        query_vector: List[float],
        k: int = 5,
        collection: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> SearchResult:
        """Search for similar vectors in Chroma.
        
        Args:
            query_vector: Vector to search for
            k: Number of results to return
            collection: Collection to search in
            filters: Metadata filters to apply
            
        Returns:
            SearchResult with matching documents and scores
        """
        if not self._initialized:
            await self.initialize()
        
        collection_name = collection or self.collection_name
        start_time = time.time()
        
        try:
            # Get collection
            chroma_collection = await self._get_collection(collection_name)
            if not chroma_collection:
                logger.warning(f"Collection '{collection_name}' not found")
                return SearchResult(
                    documents=[],
                    scores=[],
                    query="",
                    search_time=(time.time() - start_time) * 1000,
                    total_results=0,
                    collection=collection_name,
                    filters=filters
                )
            
            # Prepare where clause for filtering
            where_clause = None
            if filters:
                where_clause = self._build_where_clause(filters)
            
            # Perform search
            results = await self._run_sync(
                chroma_collection.query,
                query_embeddings=[query_vector],
                n_results=k,
                where=where_clause,
                include=["documents", "metadatas", "distances", "embeddings"]
            )
            
            # Convert results to VectorDocument objects
            documents = []
            scores = []
            
            if results and results["ids"] and results["ids"][0]:
                for i, doc_id in enumerate(results["ids"][0]):
                    # Extract data
                    content = results["documents"][0][i] if results["documents"] else ""
                    embedding = results["embeddings"][0][i] if results["embeddings"] else []
                    metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                    distance = results["distances"][0][i] if results["distances"] else 1.0
                    
                    # Convert distance to similarity score
                    # Chroma returns distances, we need similarity scores (0-1, higher = more similar)
                    if self.distance_function == "cosine":
                        # Cosine distance -> cosine similarity
                        similarity = 1.0 - distance
                    elif self.distance_function == "l2":
                        # L2 distance -> normalized similarity
                        similarity = 1.0 / (1.0 + distance)
                    elif self.distance_function == "ip":
                        # Inner product (already a similarity measure)
                        similarity = max(0.0, distance)
                    else:
                        similarity = 1.0 - distance
                    
                    # Ensure similarity is in [0, 1] range
                    similarity = max(0.0, min(1.0, similarity))
                    
                    # Extract metadata fields
                    source = metadata.pop("source", "unknown")
                    chunk_index = metadata.pop("chunk_index", 0)
                    created_at_str = metadata.pop("created_at", None)
                    
                    # Parse created_at
                    created_at = None
                    if created_at_str:
                        try:
                            from datetime import datetime
                            created_at = datetime.fromisoformat(created_at_str)
                        except (ValueError, TypeError):
                            created_at = None
                    
                    if not created_at:
                        from datetime import datetime
                        created_at = datetime.utcnow()
                    
                    # Create VectorDocument
                    vector_doc = VectorDocument(
                        id=doc_id,
                        content=content,
                        embedding=embedding,
                        metadata=metadata,
                        source=source,
                        chunk_index=int(chunk_index) if isinstance(chunk_index, (int, str)) else 0,
                        created_at=created_at
                    )
                    
                    documents.append(vector_doc)
                    scores.append(similarity)
            
            search_time = (time.time() - start_time) * 1000
            
            logger.debug(f"Chroma search returned {len(documents)} results in {search_time:.2f}ms")
            
            return SearchResult(
                documents=documents,
                scores=scores,
                query="",  # Query vector doesn't have text representation
                search_time=search_time,
                total_results=len(documents),
                collection=collection_name,
                filters=filters
            )
            
        except Exception as e:
            logger.error(f"Chroma search failed: {e}")
            raise VectorStoreError(f"Search failed: {e}") from e
    
    def _build_where_clause(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Build Chroma where clause from filters.
        
        Args:
            filters: Filter dictionary
            
        Returns:
            Chroma-compatible where clause
        """
        where_clause = {}
        
        for key, value in filters.items():
            if isinstance(value, dict):
                # Handle operators like {"$gt": 10}, {"$in": ["a", "b"]}
                where_clause[key] = value
            elif isinstance(value, list):
                # Convert list to $in operator
                where_clause[key] = {"$in": value}
            else:
                # Direct equality
                where_clause[key] = {"$eq": value}
        
        return where_clause
    
    async def delete_vectors(
        self,
        vector_ids: List[str],
        collection: Optional[str] = None
    ) -> None:
        """Delete vectors by their IDs.
        
        Args:
            vector_ids: List of vector IDs to delete
            collection: Collection to delete from
        """
        if not self._initialized:
            await self.initialize()
        
        if not vector_ids:
            return
        
        collection_name = collection or self.collection_name
        
        try:
            chroma_collection = await self._get_collection(collection_name)
            if not chroma_collection:
                logger.warning(f"Collection '{collection_name}' not found for deletion")
                return
            
            # Delete vectors
            await self._run_sync(
                chroma_collection.delete,
                ids=vector_ids
            )
            
            logger.info(f"Deleted {len(vector_ids)} vectors from collection '{collection_name}'")
            
        except Exception as e:
            logger.error(f"Failed to delete vectors from Chroma: {e}")
            raise VectorStoreError(f"Failed to delete vectors: {e}") from e
    
    async def get_collection_stats(self, collection: Optional[str] = None) -> Dict[str, Any]:
        """Get statistics about a collection.
        
        Args:
            collection: Collection name (None for default)
            
        Returns:
            Dictionary containing collection statistics
        """
        if not self._initialized:
            await self.initialize()
        
        collection_name = collection or self.collection_name
        
        try:
            chroma_collection = await self._get_collection(collection_name)
            if not chroma_collection:
                return {
                    "name": collection_name,
                    "exists": False,
                    "count": 0,
                    "dimension": None
                }
            
            # Get collection count
            count = await self._run_sync(chroma_collection.count)
            
            # Get collection metadata
            metadata = getattr(chroma_collection, "metadata", {}) or {}
            
            return {
                "name": collection_name,
                "exists": True,
                "count": count,
                "dimension": metadata.get("dimension"),
                "distance_function": self.distance_function,
                "metadata": metadata
            }
            
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            raise VectorStoreError(f"Failed to get collection stats: {e}") from e
    
    async def list_collections(self) -> List[str]:
        """List all available collections.
        
        Returns:
            List of collection names
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            collections = await self._run_sync(self.client.list_collections)
            collection_names = [col.name for col in collections]
            
            logger.debug(f"Found {len(collection_names)} collections: {collection_names}")
            return collection_names
            
        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            raise VectorStoreError(f"Failed to list collections: {e}") from e
    
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
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            # Prepare metadata
            metadata = {
                "dimension": dimension,
                "distance_function": self.distance_function
            }
            if description:
                metadata["description"] = description
            
            # Create collection
            collection = await self._run_sync(
                self.client.create_collection,
                name=name,
                metadata=metadata
            )
            
            # Cache the collection
            self.collections[name] = collection
            
            logger.info(f"Created collection '{name}' with dimension {dimension}")
            
        except Exception as e:
            logger.error(f"Failed to create collection '{name}': {e}")
            raise VectorStoreError(f"Failed to create collection: {e}") from e
    
    async def delete_collection(self, name: str) -> None:
        """Delete a collection and all its vectors.
        
        Args:
            name: Collection name to delete
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            # Delete collection
            await self._run_sync(self.client.delete_collection, name=name)
            
            # Remove from cache
            if name in self.collections:
                del self.collections[name]
            
            logger.info(f"Deleted collection '{name}'")
            
        except Exception as e:
            logger.error(f"Failed to delete collection '{name}': {e}")
            raise VectorStoreError(f"Failed to delete collection: {e}") from e
    
    async def _get_collection(self, name: str) -> Optional[ChromaCollection]:
        """Get a collection by name, with caching.
        
        Args:
            name: Collection name
            
        Returns:
            Chroma collection object or None if not found
        """
        # Check cache first
        if name in self.collections:
            return self.collections[name]
        
        try:
            # Try to get existing collection
            collection = await self._run_sync(self.client.get_collection, name=name)
            self.collections[name] = collection
            return collection
            
        except Exception:
            # Collection doesn't exist
            return None
    
    async def _get_or_create_collection(self, name: str, dimension: int) -> ChromaCollection:
        """Get existing collection or create new one.
        
        Args:
            name: Collection name
            dimension: Vector dimension
            
        Returns:
            Chroma collection object
        """
        # Try to get existing collection
        collection = await self._get_collection(name)
        if collection:
            return collection
        
        # Create new collection
        await self.create_collection(name, dimension)
        return await self._get_collection(name)
    
    async def close(self) -> None:
        """Clean up Chroma resources."""
        try:
            if self.client and hasattr(self.client, 'reset'):
                await self._run_sync(self.client.reset)
            
            self.collections.clear()
            self.client = None
            self._initialized = False
            
            logger.info("Chroma backend closed successfully")
            
        except Exception as e:
            logger.warning(f"Error closing Chroma backend: {e}")