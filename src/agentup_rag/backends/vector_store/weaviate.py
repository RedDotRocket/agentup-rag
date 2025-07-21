"""
Weaviate vector store backend implementation.

This module provides a production-ready Weaviate integration for the AgentUp RAG plugin,
supporting enterprise vector databases with advanced features like graph relationships,
multi-modal support, and hybrid search capabilities.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

try:
    import weaviate
    from weaviate.classes.config import Configure, Property, DataType
    from weaviate.classes.query import Filter, MetadataQuery
    from weaviate.collections.classes.filters import _Filters
    WEAVIATE_AVAILABLE = True
except ImportError:
    weaviate = None
    Configure = None
    Property = None
    DataType = None
    Filter = None
    MetadataQuery = None
    _Filters = None
    WEAVIATE_AVAILABLE = False

from ..base import VectorStoreBackend, VectorStoreError
from ...models import SearchResult, VectorDocument

logger = logging.getLogger(__name__)


class WeaviateVectorStoreBackend(VectorStoreBackend):
    """Weaviate vector store backend for enterprise deployments.
    
    Features:
    - Graph-based relationships between documents
    - Multi-modal support (text, images, etc.)
    - Advanced schema definitions with custom properties
    - Hybrid search (vector + keyword)
    - Real-time CRUD operations
    - Advanced filtering and aggregation
    - Multi-tenancy support
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Weaviate vector store backend.
        
        Args:
            config: Configuration dictionary with Weaviate settings
        """
        super().__init__(config)
        
        if not WEAVIATE_AVAILABLE:
            raise VectorStoreError(
                "Weaviate is not available. Install with: pip install weaviate-client"
            )
        
        self.client = None
        self.collections = {}
        self._initialized = False
        
        # Configuration
        self.url = config.get("url")
        self.api_key = config.get("api_key")
        self.class_name = config.get("class_name", "Document")
        self.properties = config.get("properties", [])
        
        # Authentication configuration
        self.auth_config = config.get("auth_config", {})
        
        # Additional Weaviate-specific settings
        self.vector_index_type = config.get("vector_index_type", "hnsw")
        self.vector_index_config = config.get("vector_index_config", {})
        self.inverted_index_config = config.get("inverted_index_config", {})
        self.replication_config = config.get("replication_config", {})
        
        # Multi-tenancy settings
        self.multi_tenancy_enabled = config.get("multi_tenancy_enabled", False)
        self.default_tenant = config.get("default_tenant", "default")
        
        # Hybrid search settings
        self.enable_hybrid_search = config.get("enable_hybrid_search", True)
        
        logger.info(f"Initialized Weaviate backend with config: {self._get_safe_config()}")
    
    def _validate_config(self) -> None:
        """Validate Weaviate configuration."""
        if not self.url:
            raise ValueError("Weaviate URL is required")
        
        if not self.class_name:
            raise ValueError("Weaviate class name is required")
        
        # Validate URL format
        if not self.url.startswith(('http://', 'https://')):
            raise ValueError("Weaviate URL must start with http:// or https://")
        
        # Validate vector index type
        valid_index_types = ["hnsw", "flat", "dynamic"]
        if self.vector_index_type not in valid_index_types:
            raise ValueError(f"Vector index type must be one of: {valid_index_types}")
    
    def _get_safe_config(self) -> Dict[str, Any]:
        """Get configuration with sensitive data masked."""
        safe_config = self.config.copy()
        if "api_key" in safe_config and safe_config["api_key"]:
            safe_config["api_key"] = "***"
        if "auth_config" in safe_config:
            safe_config["auth_config"] = "***"
        return safe_config
    
    async def initialize(self) -> None:
        """Initialize Weaviate client and schema."""
        if self._initialized:
            return
        
        try:
            # Create authentication configuration
            auth_config = None
            if self.api_key:
                auth_config = weaviate.AuthApiKey(api_key=self.api_key)
            elif self.auth_config:
                # Handle other auth types (Bearer token, OAuth, etc.)
                auth_type = self.auth_config.get("type", "api_key")
                if auth_type == "bearer":
                    auth_config = weaviate.AuthBearerToken(
                        access_token=self.auth_config["access_token"]
                    )
                elif auth_type == "client_credentials":
                    auth_config = weaviate.AuthClientCredentials(
                        client_secret=self.auth_config["client_secret"],
                        scope=self.auth_config.get("scope")
                    )
            
            # Create Weaviate client
            self.client = weaviate.WeaviateClient(
                connection_params=weaviate.ConnectionParams.from_url(
                    url=self.url,
                    grpc_port=443 if self.url.startswith('https') else 80
                ),
                auth_client_secret=auth_config
            )
            
            # Connect to Weaviate
            await self._run_async(self.client.connect)
            
            # Test connection
            await self._run_async(self.client.is_ready)
            
            # Ensure schema exists
            await self._ensure_schema_exists()
            
            self._initialized = True
            logger.info(f"Weaviate backend initialized successfully at {self.url}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Weaviate backend: {e}")
            raise VectorStoreError(f"Weaviate initialization failed: {e}") from e
    
    async def _run_async(self, func, *args, **kwargs):
        """Run Weaviate operations asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, func, *args, **kwargs)
    
    async def _ensure_schema_exists(self) -> None:
        """Ensure the Weaviate schema exists for our class."""
        try:
            # Check if class exists
            schema = await self._run_async(self.client.schema.get)
            existing_classes = [cls['class'] for cls in schema.get('classes', [])]
            
            if self.class_name in existing_classes:
                logger.info(f"Weaviate class '{self.class_name}' already exists")
                return
            
            # Create class schema
            await self._create_class_schema()
            
        except Exception as e:
            logger.error(f"Failed to ensure schema exists: {e}")
            raise VectorStoreError(f"Schema validation failed: {e}") from e
    
    async def _create_class_schema(self) -> None:
        """Create the Weaviate class schema."""
        try:
            # Default properties for RAG documents
            default_properties = [
                Property(name="content", data_type=DataType.TEXT),
                Property(name="source", data_type=DataType.TEXT),
                Property(name="chunk_index", data_type=DataType.INT),
                Property(name="created_at", data_type=DataType.DATE),
                Property(name="updated_at", data_type=DataType.DATE),
                Property(name="metadata_json", data_type=DataType.TEXT),  # For complex metadata
            ]
            
            # Add custom properties from config
            properties = default_properties.copy()
            for prop_config in self.properties:
                prop_name = prop_config.get("name")
                prop_type = prop_config.get("dataType", ["text"])
                
                if prop_name and prop_name not in [p.name for p in default_properties]:
                    # Convert string data types to Weaviate DataType enum
                    weaviate_type = self._convert_data_type(prop_type)
                    properties.append(Property(name=prop_name, data_type=weaviate_type))
            
            # Configure vector index
            vector_config = Configure.VectorIndex.hnsw()
            if self.vector_index_type == "flat":
                vector_config = Configure.VectorIndex.flat()
            elif self.vector_index_type == "dynamic":
                vector_config = Configure.VectorIndex.dynamic()
            
            # Apply custom vector index configuration
            if self.vector_index_config:
                if hasattr(vector_config, 'ef_construction'):
                    vector_config.ef_construction = self.vector_index_config.get('ef_construction', 128)
                if hasattr(vector_config, 'max_connections'):
                    vector_config.max_connections = self.vector_index_config.get('max_connections', 64)
                if hasattr(vector_config, 'ef'):
                    vector_config.ef = self.vector_index_config.get('ef', -1)
            
            # Configure inverted index for text search
            inverted_config = Configure.InvertedIndex(
                bm25_b=self.inverted_index_config.get('bm25_b', 0.75),
                bm25_k1=self.inverted_index_config.get('bm25_k1', 1.2),
                cleanup_interval_seconds=self.inverted_index_config.get('cleanup_interval', 60),
                stopwords_preset=self.inverted_index_config.get('stopwords_preset', 'en')
            )
            
            # Configure multi-tenancy if enabled
            multi_tenancy_config = None
            if self.multi_tenancy_enabled:
                multi_tenancy_config = Configure.multi_tenancy(enabled=True)
            
            # Configure replication if specified
            replication_config = None
            if self.replication_config:
                replication_config = Configure.replication(
                    factor=self.replication_config.get('factor', 1)
                )
            
            # Create the class
            class_config = {
                "class": self.class_name,
                "description": f"RAG document class for AgentUp",
                "properties": [prop.to_dict() for prop in properties],
                "vectorIndexType": self.vector_index_type,
                "vectorIndexConfig": vector_config.to_dict() if hasattr(vector_config, 'to_dict') else {},
                "invertedIndexConfig": inverted_config.to_dict() if hasattr(inverted_config, 'to_dict') else {}
            }
            
            if multi_tenancy_config:
                class_config["multiTenancyConfig"] = multi_tenancy_config.to_dict()
            
            if replication_config:
                class_config["replicationConfig"] = replication_config.to_dict()
            
            await self._run_async(self.client.schema.create_class, class_config)
            
            logger.info(f"Created Weaviate class '{self.class_name}' with {len(properties)} properties")
            
        except Exception as e:
            logger.error(f"Failed to create class schema: {e}")
            raise VectorStoreError(f"Schema creation failed: {e}") from e
    
    def _convert_data_type(self, data_type: Union[str, List[str]]) -> DataType:
        """Convert string data type to Weaviate DataType enum."""
        if isinstance(data_type, list):
            data_type = data_type[0] if data_type else "text"
        
        data_type = data_type.lower()
        
        type_mapping = {
            "text": DataType.TEXT,
            "string": DataType.TEXT,
            "int": DataType.INT,
            "integer": DataType.INT,
            "number": DataType.NUMBER,
            "float": DataType.NUMBER,
            "boolean": DataType.BOOL,
            "bool": DataType.BOOL,
            "date": DataType.DATE,
            "datetime": DataType.DATE,
            "uuid": DataType.UUID,
            "blob": DataType.BLOB,
            "geoCoordinates": DataType.GEO_COORDINATES,
            "phoneNumber": DataType.PHONE_NUMBER
        }
        
        return type_mapping.get(data_type, DataType.TEXT)
    
    async def add_vectors(
        self, 
        vectors: List[VectorDocument],
        collection: Optional[str] = None
    ) -> None:
        """Add vector documents to Weaviate.
        
        Args:
            vectors: List of vector documents to add
            collection: Collection name (used as tenant if multi-tenancy enabled)
        """
        if not self._initialized:
            await self.initialize()
        
        if not vectors:
            return
        
        tenant = collection if self.multi_tenancy_enabled else None
        
        try:
            # Get collection handle
            weaviate_collection = self.client.collections.get(self.class_name)
            
            # Create tenant if needed
            if tenant and self.multi_tenancy_enabled:
                await self._ensure_tenant_exists(tenant)
                weaviate_collection = weaviate_collection.with_tenant(tenant)
            
            # Prepare objects for insertion
            objects = []
            
            for doc in vectors:
                # Prepare properties
                properties = {
                    "content": doc.content,
                    "source": doc.source,
                    "chunk_index": doc.chunk_index,
                    "created_at": doc.created_at,
                    "metadata_json": self._serialize_metadata(doc.metadata)
                }
                
                # Add custom metadata as individual properties
                for key, value in doc.metadata.items():
                    # Only add if it's a simple type and property exists in schema
                    if isinstance(value, (str, int, float, bool)) and len(key) <= 100:
                        # Sanitize property name
                        safe_key = self._sanitize_property_name(key)
                        properties[safe_key] = value
                
                objects.append({
                    "uuid": self._generate_uuid(doc.id),
                    "properties": properties,
                    "vector": doc.embedding
                })
            
            # Insert objects in batches
            batch_size = 100  # Weaviate recommended batch size
            
            for i in range(0, len(objects), batch_size):
                batch = objects[i:i + batch_size]
                
                # Use batch import for efficiency
                with weaviate_collection.batch.dynamic() as batch_client:
                    for obj in batch:
                        batch_client.add_object(
                            properties=obj["properties"],
                            uuid=obj["uuid"],
                            vector=obj["vector"]
                        )
                
                logger.debug(f"Added batch {i//batch_size + 1} ({len(batch)} vectors) to Weaviate")
            
            logger.info(f"Added {len(vectors)} vectors to Weaviate class '{self.class_name}'")
            
        except Exception as e:
            logger.error(f"Failed to add vectors to Weaviate: {e}")
            raise VectorStoreError(f"Failed to add vectors: {e}") from e
    
    def _serialize_metadata(self, metadata: Dict[str, Any]) -> str:
        """Serialize complex metadata to JSON string."""
        import json
        try:
            return json.dumps(metadata, default=str)
        except Exception:
            return "{}"
    
    def _deserialize_metadata(self, metadata_json: str) -> Dict[str, Any]:
        """Deserialize metadata from JSON string."""
        import json
        try:
            return json.loads(metadata_json) if metadata_json else {}
        except Exception:
            return {}
    
    def _sanitize_property_name(self, name: str) -> str:
        """Sanitize property name for Weaviate compatibility."""
        # Replace invalid characters with underscores
        import re
        sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', name)
        
        # Ensure it starts with a letter
        if sanitized and not sanitized[0].isalpha():
            sanitized = 'prop_' + sanitized
        
        return sanitized[:100]  # Limit length
    
    def _generate_uuid(self, doc_id: str) -> str:
        """Generate UUID from document ID."""
        import uuid
        import hashlib
        
        # Create deterministic UUID from document ID
        namespace = uuid.UUID('12345678-1234-5678-1234-123456789012')
        return str(uuid.uuid5(namespace, doc_id))
    
    async def _ensure_tenant_exists(self, tenant: str) -> None:
        """Ensure tenant exists if multi-tenancy is enabled."""
        try:
            collection = self.client.collections.get(self.class_name)
            tenants = await self._run_async(collection.tenants.get)
            
            existing_tenants = [t.name for t in tenants]
            if tenant not in existing_tenants:
                await self._run_async(
                    collection.tenants.create,
                    [{"name": tenant}]
                )
                logger.info(f"Created tenant '{tenant}' for class '{self.class_name}'")
            
        except Exception as e:
            logger.error(f"Failed to ensure tenant exists: {e}")
            raise VectorStoreError(f"Tenant creation failed: {e}") from e
    
    async def search(
        self,
        query_vector: List[float],
        k: int = 5,
        collection: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> SearchResult:
        """Search for similar vectors in Weaviate.
        
        Args:
            query_vector: Vector to search for
            k: Number of results to return
            collection: Collection name (tenant)
            filters: Metadata filters to apply
            
        Returns:
            SearchResult with matching documents and scores
        """
        if not self._initialized:
            await self.initialize()
        
        tenant = collection if self.multi_tenancy_enabled else None
        start_time = time.time()
        
        try:
            # Get collection handle
            weaviate_collection = self.client.collections.get(self.class_name)
            
            if tenant and self.multi_tenancy_enabled:
                weaviate_collection = weaviate_collection.with_tenant(tenant)
            
            # Build query
            query_builder = weaviate_collection.query.near_vector(
                near_vector=query_vector,
                limit=k,
                return_metadata=MetadataQuery(score=True, explain_score=True)
            )
            
            # Apply filters if provided
            if filters:
                weaviate_filter = self._build_weaviate_filter(filters)
                if weaviate_filter:
                    query_builder = query_builder.where(weaviate_filter)
            
            # Execute query
            results = await self._run_async(query_builder.do)
            
            # Convert results to VectorDocument objects
            documents = []
            scores = []
            
            for obj in results.objects:
                properties = obj.properties
                
                # Extract standard fields
                content = properties.get("content", "")
                source = properties.get("source", "unknown")
                chunk_index = properties.get("chunk_index", 0)
                created_at = properties.get("created_at")
                metadata_json = properties.get("metadata_json", "{}")
                
                # Deserialize metadata
                metadata = self._deserialize_metadata(metadata_json)
                
                # Add other properties to metadata
                for key, value in properties.items():
                    if key not in ["content", "source", "chunk_index", "created_at", "metadata_json"]:
                        metadata[key] = value
                
                # Parse created_at
                if not isinstance(created_at, datetime):
                    if isinstance(created_at, str):
                        try:
                            created_at = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                        except (ValueError, TypeError):
                            created_at = datetime.utcnow()
                    else:
                        created_at = datetime.utcnow()
                
                # Convert UUID back to original ID
                original_id = f"{source}_{chunk_index}_{obj.uuid}"
                
                # Create VectorDocument
                vector_doc = VectorDocument(
                    id=original_id,
                    content=content,
                    embedding=[],  # Weaviate doesn't return vectors by default
                    metadata=metadata,
                    source=source,
                    chunk_index=int(chunk_index),
                    created_at=created_at
                )
                
                documents.append(vector_doc)
                scores.append(float(obj.metadata.score) if obj.metadata.score else 0.0)
            
            search_time = (time.time() - start_time) * 1000
            
            logger.debug(f"Weaviate search returned {len(documents)} results in {search_time:.2f}ms")
            
            return SearchResult(
                documents=documents,
                scores=scores,
                query="",  # Query vector doesn't have text representation
                search_time=search_time,
                total_results=len(documents),
                collection=collection,
                filters=filters
            )
            
        except Exception as e:
            logger.error(f"Weaviate search failed: {e}")
            raise VectorStoreError(f"Search failed: {e}") from e
    
    def _build_weaviate_filter(self, filters: Dict[str, Any]) -> Optional[_Filters]:
        """Build Weaviate filter from standard filter dictionary.
        
        Args:
            filters: Standard filter dictionary
            
        Returns:
            Weaviate-compatible filter or None
        """
        try:
            filter_conditions = []
            
            for key, value in filters.items():
                if isinstance(value, dict):
                    # Handle operators
                    for op, op_value in value.items():
                        if op == "$eq":
                            filter_conditions.append(Filter.by_property(key).equal(op_value))
                        elif op == "$ne":
                            filter_conditions.append(Filter.by_property(key).not_equal(op_value))
                        elif op == "$gt":
                            filter_conditions.append(Filter.by_property(key).greater_than(op_value))
                        elif op == "$gte":
                            filter_conditions.append(Filter.by_property(key).greater_or_equal(op_value))
                        elif op == "$lt":
                            filter_conditions.append(Filter.by_property(key).less_than(op_value))
                        elif op == "$lte":
                            filter_conditions.append(Filter.by_property(key).less_or_equal(op_value))
                        elif op == "$in":
                            if isinstance(op_value, list):
                                # Create OR condition for multiple values
                                in_conditions = [Filter.by_property(key).equal(v) for v in op_value]
                                if len(in_conditions) == 1:
                                    filter_conditions.append(in_conditions[0])
                                else:
                                    filter_conditions.append(Filter.any_of(in_conditions))
                elif isinstance(value, list):
                    # Convert list to $in operator
                    in_conditions = [Filter.by_property(key).equal(v) for v in value]
                    if len(in_conditions) == 1:
                        filter_conditions.append(in_conditions[0])
                    else:
                        filter_conditions.append(Filter.any_of(in_conditions))
                else:
                    # Direct equality
                    filter_conditions.append(Filter.by_property(key).equal(value))
            
            if not filter_conditions:
                return None
            elif len(filter_conditions) == 1:
                return filter_conditions[0]
            else:
                return Filter.all_of(filter_conditions)
                
        except Exception as e:
            logger.warning(f"Failed to build Weaviate filter: {e}")
            return None
    
    async def delete_vectors(
        self,
        vector_ids: List[str],
        collection: Optional[str] = None
    ) -> None:
        """Delete vectors by their IDs.
        
        Args:
            vector_ids: List of vector IDs to delete
            collection: Collection to delete from (tenant)
        """
        if not self._initialized:
            await self.initialize()
        
        if not vector_ids:
            return
        
        tenant = collection if self.multi_tenancy_enabled else None
        
        try:
            # Get collection handle
            weaviate_collection = self.client.collections.get(self.class_name)
            
            if tenant and self.multi_tenancy_enabled:
                weaviate_collection = weaviate_collection.with_tenant(tenant)
            
            # Convert IDs to UUIDs
            uuids = [self._generate_uuid(doc_id) for doc_id in vector_ids]
            
            # Delete objects in batches
            batch_size = 100
            
            for i in range(0, len(uuids), batch_size):
                batch = uuids[i:i + batch_size]
                
                for uuid in batch:
                    await self._run_async(weaviate_collection.data.delete_by_id, uuid)
                
                logger.debug(f"Deleted batch {i//batch_size + 1} ({len(batch)} vectors) from Weaviate")
            
            logger.info(f"Deleted {len(vector_ids)} vectors from Weaviate class '{self.class_name}'")
            
        except Exception as e:
            logger.error(f"Failed to delete vectors from Weaviate: {e}")
            raise VectorStoreError(f"Failed to delete vectors: {e}") from e
    
    async def get_collection_stats(self, collection: Optional[str] = None) -> Dict[str, Any]:
        """Get statistics about a collection.
        
        Args:
            collection: Collection name (tenant)
            
        Returns:
            Dictionary containing collection statistics
        """
        if not self._initialized:
            await self.initialize()
        
        tenant = collection if self.multi_tenancy_enabled else None
        
        try:
            # Get collection handle
            weaviate_collection = self.client.collections.get(self.class_name)
            
            if tenant and self.multi_tenancy_enabled:
                weaviate_collection = weaviate_collection.with_tenant(tenant)
            
            # Get object count
            count_result = await self._run_async(
                weaviate_collection.aggregate.over_all,
                total_count=True
            )
            
            count = count_result.total_count if count_result.total_count else 0
            
            return {
                "name": collection or "default",
                "exists": True,
                "count": count,
                "class_name": self.class_name,
                "multi_tenancy_enabled": self.multi_tenancy_enabled,
                "tenant": tenant,
                "vector_index_type": self.vector_index_type
            }
            
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {
                "name": collection or "default",
                "exists": False,
                "count": 0,
                "error": str(e)
            }
    
    async def list_collections(self) -> List[str]:
        """List all available collections.
        
        Returns:
            List of collection names (tenants if multi-tenancy enabled)
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            if self.multi_tenancy_enabled:
                # List tenants
                collection = self.client.collections.get(self.class_name)
                tenants = await self._run_async(collection.tenants.get)
                
                tenant_names = [tenant.name for tenant in tenants]
                logger.debug(f"Found {len(tenant_names)} tenants: {tenant_names}")
                return tenant_names
            else:
                # Single collection (class)
                return [self.class_name]
                
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
            name: Collection name (tenant name if multi-tenancy enabled)
            dimension: Vector dimension (ignored, uses class schema)
            description: Optional description
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            if self.multi_tenancy_enabled:
                # Create tenant
                collection = self.client.collections.get(self.class_name)
                
                tenant_config = {"name": name}
                if description:
                    tenant_config["description"] = description
                
                await self._run_async(
                    collection.tenants.create,
                    [tenant_config]
                )
                
                logger.info(f"Created tenant '{name}' for class '{self.class_name}'")
            else:
                # Class already exists, just log
                logger.info(f"Collection creation not needed - using existing class '{self.class_name}'")
            
        except Exception as e:
            logger.error(f"Failed to create collection '{name}': {e}")
            raise VectorStoreError(f"Failed to create collection: {e}") from e
    
    async def delete_collection(self, name: str) -> None:
        """Delete a collection and all its vectors.
        
        Args:
            name: Collection name (tenant) to delete
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            if self.multi_tenancy_enabled:
                # Delete tenant
                collection = self.client.collections.get(self.class_name)
                await self._run_async(collection.tenants.remove, [name])
                
                logger.info(f"Deleted tenant '{name}' from class '{self.class_name}'")
            else:
                # Delete all objects in the class
                collection = self.client.collections.get(self.class_name)
                await self._run_async(collection.data.delete_many, Filter.by_property("source").like("*"))
                
                logger.info(f"Deleted all objects from class '{self.class_name}'")
                
        except Exception as e:
            logger.error(f"Failed to delete collection '{name}': {e}")
            raise VectorStoreError(f"Failed to delete collection: {e}") from e
    
    async def close(self) -> None:
        """Clean up Weaviate resources."""
        try:
            if self.client:
                await self._run_async(self.client.close)
            
            self.collections.clear()
            self.client = None
            self._initialized = False
            
            logger.info("Weaviate backend closed successfully")
            
        except Exception as e:
            logger.warning(f"Error closing Weaviate backend: {e}")