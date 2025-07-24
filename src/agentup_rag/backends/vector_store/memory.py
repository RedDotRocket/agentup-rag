"""
Memory vector store backend implementation.

This module provides an in-memory vector store backend using FAISS for efficient
similarity search. It's ideal for development, testing, and small-scale deployments.
"""

import asyncio
import logging
import pickle  # nosec
import time
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from ...models import SearchResult, VectorDocument
from ..base import VectorStoreBackend, VectorStoreError

logger = logging.getLogger(__name__)


class MemoryVectorStoreBackend(VectorStoreBackend):
    """In-memory vector store backend with FAISS integration.

    This backend stores vectors in memory using FAISS for efficient similarity search.
    It supports different index types and can optionally persist data to disk.

    Features:
    - Multiple FAISS index types (Flat, IVF, HNSW)
    - Collection management
    - Metadata filtering
    - Optional persistence to disk
    - Automatic saving with configurable intervals
    """

    def __init__(self, config: dict[str, Any]):
        """Initialize the memory vector store backend.

        Args:
            config: Configuration dictionary with memory store settings
        """
        super().__init__(config)

        self.similarity_metric = config.get("similarity_metric", "cosine")
        self.index_type = config.get("index_type", "flat")
        self.persist_path = config.get("persist_path")
        self.auto_save = config.get("auto_save", True)
        self.save_interval = config.get("save_interval", 300)  # 5 minutes

        # Internal state
        self._collections: dict[str, dict[str, Any]] = {}
        self._initialized = False
        self._save_task: asyncio.Task | None = None
        self._lock = asyncio.Lock()

        # Validate similarity metric
        if self.similarity_metric not in ["cosine", "euclidean", "dot_product"]:
            raise ValueError(f"Unsupported similarity metric: {self.similarity_metric}")

        logger.info(f"Initialized memory vector store with {self.index_type} index")

    def _validate_config(self) -> None:
        """Validate the memory store configuration.

        Raises:
            ValueError: If configuration is invalid
        """
        index_type = self.config.get("index_type", "flat")
        if index_type not in ["flat", "ivf", "hnsw"]:
            raise ValueError(f"Unsupported index type: {index_type}")

        save_interval = self.config.get("save_interval", 300)
        if save_interval <= 0:
            raise ValueError("Save interval must be positive")

    async def initialize(self) -> None:
        """Initialize the vector store backend.

        This loads persisted data if available and sets up auto-saving.
        """
        async with self._lock:
            if self._initialized:
                return

            # Load persisted data if available
            if self.persist_path and Path(self.persist_path).exists():
                await self._load_from_disk()

            # Start auto-save task if enabled
            if self.auto_save and self.persist_path:
                self._save_task = asyncio.create_task(self._auto_save_loop())

            self._initialized = True
            logger.info("Memory vector store backend initialized")

    def _create_faiss_index(self, dimension: int) -> faiss.Index:
        """Create a FAISS index based on configuration.

        Args:
            dimension: Vector dimension

        Returns:
            FAISS index instance
        """
        if self.similarity_metric == "cosine":
            # For cosine similarity, we'll normalize vectors and use inner product
            if self.index_type == "flat":
                index = faiss.IndexFlatIP(dimension)
            elif self.index_type == "ivf":
                quantizer = faiss.IndexFlatIP(dimension)
                index = faiss.IndexIVFFlat(quantizer, dimension, 100)  # 100 clusters
                index.nprobe = 10  # Search 10 clusters
            elif self.index_type == "hnsw":
                index = faiss.IndexHNSWFlat(dimension, 32)  # 32 connections per node
                index.hnsw.efConstruction = 200
                index.hnsw.efSearch = 50
            else:
                raise ValueError(f"Unsupported index type: {self.index_type}")

        elif self.similarity_metric == "euclidean":
            if self.index_type == "flat":
                index = faiss.IndexFlatL2(dimension)
            elif self.index_type == "ivf":
                quantizer = faiss.IndexFlatL2(dimension)
                index = faiss.IndexIVFFlat(quantizer, dimension, 100)
                index.nprobe = 10
            elif self.index_type == "hnsw":
                index = faiss.IndexHNSWFlat(dimension, 32)
            else:
                raise ValueError(f"Unsupported index type: {self.index_type}")

        elif self.similarity_metric == "dot_product":
            if self.index_type == "flat":
                index = faiss.IndexFlatIP(dimension)
            elif self.index_type == "ivf":
                quantizer = faiss.IndexFlatIP(dimension)
                index = faiss.IndexIVFFlat(quantizer, dimension, 100)
                index.nprobe = 10
            else:
                # HNSW doesn't directly support inner product, use flat
                index = faiss.IndexFlatIP(dimension)

        return index

    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """Normalize vectors for cosine similarity.

        Args:
            vectors: Array of vectors to normalize

        Returns:
            Normalized vectors
        """
        if self.similarity_metric == "cosine":
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            # Avoid division by zero
            norms = np.where(norms == 0, 1, norms)
            return vectors / norms
        return vectors

    def _create_collection_unlocked(self, name: str, dimension: int, description: str | None = None) -> None:
        """Create a new collection without acquiring lock.

        Args:
            name: Collection name
            dimension: Vector dimension for this collection
            description: Optional collection description
        """
        if name in self._collections:
            raise VectorStoreError(f"Collection '{name}' already exists")

        index = self._create_faiss_index(dimension)

        # Train IVF index if necessary
        if hasattr(index, "is_trained") and not index.is_trained:
            # For IVF, we need some training data
            # Create dummy training data if needed
            training_data = np.random.random((max(1000, 100), dimension)).astype("float32")
            training_data = self._normalize_vectors(training_data)
            index.train(training_data)

        self._collections[name] = {
            "index": index,
            "documents": {},  # id -> VectorDocument
            "id_to_index": {},  # document_id -> faiss_index
            "index_to_id": {},  # faiss_index -> document_id
            "dimension": dimension,
            "description": description or "",
            "created_at": time.time(),
            "document_count": 0,
        }

        logger.info(f"Created collection '{name}' with dimension {dimension}")

    async def create_collection(self, name: str, dimension: int, description: str | None = None) -> None:
        """Create a new collection.

        Args:
            name: Collection name
            dimension: Vector dimension for this collection
            description: Optional collection description
        """
        async with self._lock:
            self._create_collection_unlocked(name, dimension, description)

    async def delete_collection(self, name: str) -> None:
        """Delete a collection and all its vectors.

        Args:
            name: Collection name to delete
        """
        async with self._lock:
            if name not in self._collections:
                raise VectorStoreError(f"Collection '{name}' does not exist")

            del self._collections[name]
            logger.info(f"Deleted collection '{name}'")

    async def list_collections(self) -> list[str]:
        """list all available collections.

        Returns:
            list of collection names
        """
        return list(self._collections.keys())

    async def get_collection_stats(self, collection: str | None = None) -> dict[str, Any]:
        """Get statistics about a collection.

        Args:
            collection: Collection name (None for default)

        Returns:
            dictionary containing collection statistics
        """
        collection_name = collection or "default"

        if collection_name not in self._collections:
            return {
                "exists": False,
                "name": collection_name,
            }

        coll = self._collections[collection_name]

        return {
            "exists": True,
            "name": collection_name,
            "description": coll["description"],
            "dimension": coll["dimension"],
            "document_count": coll["document_count"],
            "index_type": self.index_type,
            "similarity_metric": self.similarity_metric,
            "created_at": coll["created_at"],
            "total_vectors": coll["index"].ntotal,
        }

    async def add_vectors(self, vectors: list[VectorDocument], collection: str | None = None) -> None:
        """Add vector documents to the store.

        Args:
            vectors: list of vector documents to add
            collection: Collection name (creates default if None)
        """
        if not vectors:
            return

        collection_name = collection or "default"

        async with self._lock:
            # Create collection if it doesn't exist
            if collection_name not in self._collections:
                # Infer dimension from first vector
                dimension = len(vectors[0].embedding)
                self._create_collection_unlocked(collection_name, dimension)

            coll = self._collections[collection_name]

            # Validate dimensions
            expected_dim = coll["dimension"]
            for vector in vectors:
                if len(vector.embedding) != expected_dim:
                    raise VectorStoreError(
                        f"Vector dimension {len(vector.embedding)} doesn't match collection dimension {expected_dim}"
                    )

            # Convert embeddings to numpy array
            embeddings = np.array([v.embedding for v in vectors], dtype="float32")
            embeddings = self._normalize_vectors(embeddings)

            # Get current index size before adding
            current_size = coll["index"].ntotal

            # Add to index
            coll["index"].add(embeddings)

            # Store documents and maintain mappings
            for i, vector in enumerate(vectors):
                faiss_idx = current_size + i
                coll["documents"][vector.id] = vector
                coll["id_to_index"][vector.id] = faiss_idx
                coll["index_to_id"][faiss_idx] = vector.id

            # Update stats
            coll["document_count"] += len(vectors)

            logger.debug(f"Added {len(vectors)} vectors to collection '{collection_name}'")

    async def search(
        self,
        query_vector: list[float],
        k: int = 5,
        collection: str | None = None,
        filters: dict[str, Any] | None = None,
    ) -> SearchResult:
        """Search for similar vectors.

        Args:
            query_vector: Vector to search for
            k: Number of results to return
            collection: Collection to search in
            filters: Metadata filters to apply

        Returns:
            SearchResult containing matching documents and scores
        """
        collection_name = collection or "default"

        if collection_name not in self._collections:
            raise VectorStoreError(f"Collection '{collection_name}' does not exist")

        coll = self._collections[collection_name]

        if coll["document_count"] == 0:
            return SearchResult(
                documents=[],
                scores=[],
                query="",
                search_time=0.0,
                total_results=0,
                collection=collection_name,
                filters=filters,
            )

        start_time = time.time()

        # Prepare query vector
        query_array = np.array([query_vector], dtype="float32")
        query_array = self._normalize_vectors(query_array)

        # Search the index
        search_k = min(k * 2, coll["index"].ntotal)  # Get more results for filtering
        scores, indices = coll["index"].search(query_array, search_k)

        # Process results
        documents = []
        result_scores = []

        for i, idx in enumerate(indices[0]):
            if idx == -1:  # FAISS returns -1 for empty slots
                break

            score = float(scores[0][i])

            # Convert FAISS score based on similarity metric
            if self.similarity_metric == "euclidean":
                # For L2 distance, smaller is better, so invert
                score = 1.0 / (1.0 + score)
            elif self.similarity_metric in ["cosine", "dot_product"]:
                # For inner product, ensure score is between 0 and 1
                score = max(0.0, min(1.0, score))

            # Find document by FAISS index using our mapping
            if idx in coll["index_to_id"]:
                doc_id = coll["index_to_id"][idx]
                document = coll["documents"][doc_id]

                # Apply metadata filters if specified
                if filters:
                    if not self._matches_filters(document, filters):
                        continue

                documents.append(document)
                result_scores.append(score)

                if len(documents) >= k:
                    break

        search_time = (time.time() - start_time) * 1000  # Convert to milliseconds

        return SearchResult(
            documents=documents,
            scores=result_scores,
            query="",  # We don't store the original query text
            search_time=search_time,
            total_results=len(documents),
            collection=collection_name,
            filters=filters,
        )

    def _matches_filters(self, document: VectorDocument, filters: dict[str, Any]) -> bool:
        """Check if a document matches the given filters.

        Args:
            document: Document to check
            filters: Filters to apply

        Returns:
            True if document matches all filters
        """
        for key, value in filters.items():
            doc_value = document.metadata.get(key)

            if isinstance(value, dict):
                # Handle operators like {"$gt": 10}
                for op, op_value in value.items():
                    if op == "$gt" and not (doc_value > op_value):
                        return False
                    elif op == "$gte" and not (doc_value >= op_value):
                        return False
                    elif op == "$lt" and not (doc_value < op_value):
                        return False
                    elif op == "$lte" and not (doc_value <= op_value):
                        return False
                    elif op == "$ne" and not (doc_value != op_value):
                        return False
                    elif op == "$in" and doc_value not in op_value:
                        return False
            else:
                # Direct equality
                if doc_value != value:
                    return False

        return True

    async def delete_vectors(self, vector_ids: list[str], collection: str | None = None) -> None:
        """Delete vectors by their IDs.

        Note: This is inefficient in FAISS as it requires rebuilding the index.
        For production use, consider other vector stores with better deletion support.

        Args:
            vector_ids: list of vector IDs to delete
            collection: Collection to delete from
        """
        collection_name = collection or "default"

        if collection_name not in self._collections:
            raise VectorStoreError(f"Collection '{collection_name}' does not exist")

        async with self._lock:
            coll = self._collections[collection_name]

            # Remove documents and clear mappings
            removed_count = 0
            for vector_id in vector_ids:
                if vector_id in coll["documents"]:
                    # Remove from mappings
                    if vector_id in coll["id_to_index"]:
                        old_idx = coll["id_to_index"][vector_id]
                        del coll["id_to_index"][vector_id]
                        if old_idx in coll["index_to_id"]:
                            del coll["index_to_id"][old_idx]
                    # Remove document
                    del coll["documents"][vector_id]
                    removed_count += 1

            # Rebuild index (this is expensive but necessary for FAISS)
            if removed_count > 0:
                remaining_docs = list(coll["documents"].values())

                # Create new index
                new_index = self._create_faiss_index(coll["dimension"])

                # Clear and rebuild mappings
                coll["id_to_index"] = {}
                coll["index_to_id"] = {}

                if remaining_docs:
                    embeddings = np.array([d.embedding for d in remaining_docs], dtype="float32")
                    embeddings = self._normalize_vectors(embeddings)
                    new_index.add(embeddings)

                    # Rebuild mappings
                    for i, doc in enumerate(remaining_docs):
                        coll["id_to_index"][doc.id] = i
                        coll["index_to_id"][i] = doc.id

                coll["index"] = new_index
                coll["document_count"] = len(remaining_docs)

                logger.info(f"Deleted {removed_count} vectors from collection '{collection_name}'")

    async def _save_to_disk(self) -> None:
        """Save the current state to disk."""
        if not self.persist_path:
            return

        try:
            persist_path = Path(self.persist_path)
            persist_path.parent.mkdir(parents=True, exist_ok=True)

            # Prepare data for serialization
            save_data = {
                "collections": {},
                "config": {
                    "similarity_metric": self.similarity_metric,
                    "index_type": self.index_type,
                },
            }

            for name, coll in self._collections.items():
                # Save index
                index_path = persist_path.parent / f"{persist_path.stem}_{name}_index.faiss"
                faiss.write_index(coll["index"], str(index_path))

                # Save collection metadata and documents
                save_data["collections"][name] = {
                    "dimension": coll["dimension"],
                    "description": coll["description"],
                    "created_at": coll["created_at"],
                    "document_count": coll["document_count"],
                    "index_path": str(index_path),
                    "id_to_index": coll["id_to_index"],
                    "index_to_id": coll["index_to_id"],
                    "documents": {
                        doc_id: {
                            "id": doc.id,
                            "content": doc.content,
                            "embedding": doc.embedding,
                            "metadata": doc.metadata,
                            "source": doc.source,
                            "chunk_index": doc.chunk_index,
                            "created_at": doc.created_at.isoformat(),
                            "updated_at": doc.updated_at.isoformat() if doc.updated_at else None,
                        }
                        for doc_id, doc in coll["documents"].items()
                    },
                }

            # Save to file
            with open(persist_path, "wb") as f:
                pickle.dump(save_data, f)

            logger.debug(f"Saved memory vector store to {persist_path}")

        except Exception as e:
            logger.error(f"Failed to save memory vector store: {e}")

    async def _load_from_disk(self) -> None:
        """
        Load state from disk.
        Bandit: B301. I need to come back to this and ensure secure loading.
        This will load collections, indices, and documents from the saved file.
        """
        try:
            with open(self.persist_path, "rb") as f:
                save_data = pickle.load(f)  # nosec: B301

            # Load collections
            for name, coll_data in save_data["collections"].items():
                # Load index
                index_path = coll_data["index_path"]
                if Path(index_path).exists():
                    index = faiss.read_index(index_path)
                else:
                    # Create empty index if file doesn't exist
                    index = self._create_faiss_index(coll_data["dimension"])

                # Recreate documents
                documents = {}
                for doc_id, doc_data in coll_data["documents"].items():
                    from datetime import datetime

                    documents[doc_id] = VectorDocument(
                        id=doc_data["id"],
                        content=doc_data["content"],
                        embedding=doc_data["embedding"],
                        metadata=doc_data["metadata"],
                        source=doc_data["source"],
                        chunk_index=doc_data["chunk_index"],
                        created_at=datetime.fromisoformat(doc_data["created_at"]),
                        updated_at=datetime.fromisoformat(doc_data["updated_at"]) if doc_data["updated_at"] else None,
                    )

                self._collections[name] = {
                    "index": index,
                    "documents": documents,
                    "id_to_index": coll_data.get("id_to_index", {}),
                    "index_to_id": coll_data.get("index_to_id", {}),
                    "dimension": coll_data["dimension"],
                    "description": coll_data["description"],
                    "created_at": coll_data["created_at"],
                    "document_count": coll_data["document_count"],
                }

            logger.info(f"Loaded memory vector store from {self.persist_path}")

        except Exception as e:
            logger.error(f"Failed to load memory vector store: {e}")

    async def _auto_save_loop(self) -> None:
        """Auto-save loop that runs in the background."""
        while True:
            try:
                await asyncio.sleep(self.save_interval)
                await self._save_to_disk()
            except asyncio.CancelledError:
                # Final save before shutdown
                await self._save_to_disk()
                break
            except Exception as e:
                logger.error(f"Error in auto-save loop: {e}")

    async def close(self) -> None:
        """Clean up resources used by the backend."""
        # Cancel auto-save task
        if self._save_task:
            self._save_task.cancel()
            try:
                await self._save_task
            except asyncio.CancelledError:
                pass

        # Final save
        if self.persist_path:
            await self._save_to_disk()

        logger.debug("Closed memory vector store backend")
