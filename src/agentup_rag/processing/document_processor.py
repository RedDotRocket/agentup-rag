"""
Document processor for coordinating document processing operations.

This module provides a high-level interface for processing documents,
including loading, chunking, and preparing them for embedding.
"""

import logging
import time

from ..models import Chunk, DocumentProcessingResult
from .chunking import TextChunker

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """High-level document processor.

    Coordinates various document processing operations including
    loading, chunking, and content preparation.
    """

    def __init__(self):
        """Initialize the document processor."""
        self.chunker: TextChunker | None = None
        logger.info("Initialized document processor")

    def set_chunker(self, chunker: TextChunker) -> None:
        """Set the text chunker to use.

        Args:
            chunker: Text chunker instance
        """
        self.chunker = chunker
        logger.debug("Set text chunker for document processor")

    def process_text(self, content: str, source: str, metadata: dict | None = None) -> DocumentProcessingResult:
        """Process text content into chunks.

        Args:
            content: Text content to process
            source: Source identifier
            metadata: Optional metadata to include

        Returns:
            DocumentProcessingResult with processing outcome
        """
        start_time = time.time()

        try:
            if not content or not content.strip():
                return DocumentProcessingResult(
                    success=False,
                    chunks_created=0,
                    source=source,
                    processing_time=0.0,
                    error="Empty content provided",
                )

            # Preprocess content
            processed_content = self._preprocess_content(content)

            # Chunk the content if chunker is available
            if self.chunker:
                chunks = self.chunker.chunk_text(processed_content, source)

                # Enhance chunks with additional metadata
                if metadata:
                    for chunk in chunks:
                        chunk.metadata.update(metadata)

            else:
                # Create a single chunk if no chunker is configured
                chunks = [
                    Chunk(
                        content=processed_content,
                        chunk_index=0,
                        metadata={
                            "source": source,
                            **(metadata or {}),
                        },
                    )
                ]

            processing_time = (time.time() - start_time) * 1000

            logger.debug(f"Processed document '{source}' into {len(chunks)} chunks")

            return DocumentProcessingResult(
                success=True,
                chunks_created=len(chunks),
                source=source,
                processing_time=processing_time,
                metadata={
                    "original_length": len(content),
                    "processed_length": len(processed_content),
                    "chunks": [
                        {
                            "index": chunk.chunk_index,
                            "length": len(chunk.content),
                            "start_char": chunk.start_char,
                            "end_char": chunk.end_char,
                        }
                        for chunk in chunks
                    ],
                },
            )

        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            logger.error(f"Document processing failed for '{source}': {e}")

            return DocumentProcessingResult(
                success=False,
                chunks_created=0,
                source=source,
                processing_time=processing_time,
                error=str(e),
            )

    def _preprocess_content(self, content: str) -> str:
        """Preprocess content before chunking.

        Args:
            content: Raw content

        Returns:
            Preprocessed content
        """
        # Normalize whitespace
        content = " ".join(content.split())

        # Remove excessive line breaks
        content = content.replace("\n\n\n", "\n\n")

        # Handle common encoding issues
        replacements = {
            "\u2018": "'",  # Left single quotation mark
            "\u2019": "'",  # Right single quotation mark
            "\u201c": '"',  # Left double quotation mark
            "\u201d": '"',  # Right double quotation mark
            "\u2013": "-",  # En dash
            "\u2014": "--",  # Em dash
            "\u2026": "...",  # Horizontal ellipsis
        }

        for old, new in replacements.items():
            content = content.replace(old, new)

        return content.strip()

    def estimate_processing_time(self, content: str) -> float:
        """Estimate processing time for content.

        Args:
            content: Content to estimate for

        Returns:
            Estimated processing time in seconds
        """
        # Simple estimation based on content length
        # Roughly 10ms per 1000 characters
        base_time = len(content) / 1000 * 0.01

        # Add overhead for chunking operations
        chunking_overhead = 0.05

        return base_time + chunking_overhead

    def get_processing_stats(self) -> dict:
        """Get processing statistics.

        Returns:
            Dictionary with processing statistics
        """
        return {
            "processor_type": "basic",
            "chunker_configured": self.chunker is not None,
            "chunker_strategy": self.chunker.config.strategy if self.chunker else None,
        }
