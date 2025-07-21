"""
Text chunking implementations for document processing.

This module provides various text chunking strategies for breaking down
large documents into smaller, manageable pieces for embedding and search.
"""

import logging
import re
from typing import List

from ..models import Chunk, ChunkingConfig, ChunkingStrategy

logger = logging.getLogger(__name__)


class TextChunker:
    """Text chunking utility with multiple strategies.
    
    Supports fixed-size, recursive, and semantic chunking approaches
    for optimal document segmentation.
    """
    
    def __init__(self, config: ChunkingConfig):
        """Initialize the text chunker.
        
        Args:
            config: Chunking configuration
        """
        self.config = config
        logger.info(f"Initialized text chunker with {config.strategy} strategy")
    
    def chunk_text(self, text: str, source: str = "") -> List[Chunk]:
        """Chunk text using the configured strategy.
        
        Args:
            text: Text to chunk
            source: Source identifier for metadata
            
        Returns:
            List of text chunks
        """
        if not text or not text.strip():
            return []
        
        text = text.strip()
        
        if self.config.strategy == ChunkingStrategy.FIXED:
            return self._chunk_fixed_size(text, source)
        elif self.config.strategy == ChunkingStrategy.RECURSIVE:
            return self._chunk_recursive(text, source)
        elif self.config.strategy == ChunkingStrategy.SEMANTIC:
            return self._chunk_semantic(text, source)
        else:
            raise ValueError(f"Unknown chunking strategy: {self.config.strategy}")
    
    def _chunk_fixed_size(self, text: str, source: str) -> List[Chunk]:
        """Chunk text using fixed-size strategy.
        
        Args:
            text: Text to chunk
            source: Source identifier
            
        Returns:
            List of fixed-size chunks
        """
        chunks = []
        chunk_index = 0
        
        for i in range(0, len(text), self.config.chunk_size - self.config.chunk_overlap):
            start = i
            end = min(i + self.config.chunk_size, len(text))
            
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunks.append(Chunk(
                    content=chunk_text,
                    chunk_index=chunk_index,
                    metadata={
                        "source": source,
                        "strategy": "fixed",
                        "start_char": start,
                        "end_char": end,
                    },
                    start_char=start,
                    end_char=end,
                ))
                chunk_index += 1
        
        logger.debug(f"Created {len(chunks)} fixed-size chunks from text")
        return chunks
    
    def _chunk_recursive(self, text: str, source: str) -> List[Chunk]:
        """Chunk text using recursive strategy with separators.
        
        This strategy tries to split on natural boundaries like paragraphs,
        sentences, etc., while respecting size limits.
        
        Args:
            text: Text to chunk
            source: Source identifier
            
        Returns:
            List of recursively chunked text
        """
        chunks = []
        chunk_index = 0
        
        def split_text(text: str, separators: List[str]) -> List[str]:
            """Recursively split text using separators."""
            if not separators or len(text) <= self.config.chunk_size:
                return [text]
            
            separator = separators[0]
            remaining_separators = separators[1:]
            
            # Split by current separator
            parts = text.split(separator)
            
            result = []
            current_chunk = ""
            
            for part in parts:
                # Add separator back (except for the first part)
                if current_chunk:
                    test_chunk = current_chunk + separator + part
                else:
                    test_chunk = part
                
                if len(test_chunk) <= self.config.chunk_size:
                    current_chunk = test_chunk
                else:
                    # Current chunk is good, start new one
                    if current_chunk:
                        result.append(current_chunk)
                    
                    # If part is too large, recursively split it
                    if len(part) > self.config.chunk_size:
                        result.extend(split_text(part, remaining_separators))
                        current_chunk = ""
                    else:
                        current_chunk = part
            
            # Add remaining chunk
            if current_chunk:
                result.append(current_chunk)
            
            return result
        
        # Split the text
        text_parts = split_text(text, self.config.separators)
        
        # Create chunks with overlap
        for i, part in enumerate(text_parts):
            # Calculate character positions
            start_char = 0
            for j in range(i):
                start_char += len(text_parts[j])
            
            end_char = start_char + len(part)
            
            # Add overlap from previous chunk if configured
            chunk_start = start_char
            if i > 0 and self.config.chunk_overlap > 0:
                # Find overlap text from previous chunk
                overlap_start = max(0, start_char - self.config.chunk_overlap)
                overlap_text = text[overlap_start:start_char]
                part = overlap_text + part
                chunk_start = overlap_start
            
            chunk_text = part.strip()
            
            if chunk_text:
                chunks.append(Chunk(
                    content=chunk_text,
                    chunk_index=chunk_index,
                    metadata={
                        "source": source,
                        "strategy": "recursive",
                        "start_char": chunk_start,
                        "end_char": end_char,
                    },
                    start_char=chunk_start,
                    end_char=end_char,
                ))
                chunk_index += 1
        
        logger.debug(f"Created {len(chunks)} recursive chunks from text")
        return chunks
    
    def _chunk_semantic(self, text: str, source: str) -> List[Chunk]:
        """Chunk text using semantic strategy.
        
        This strategy groups sentences with similar semantic meaning,
        requiring an embedding model for similarity computation.
        
        Args:
            text: Text to chunk
            source: Source identifier
            
        Returns:
            List of semantically coherent chunks
        """
        # For now, fall back to recursive chunking
        # In a full implementation, this would use sentence embeddings
        # to group semantically similar sentences together
        logger.warning("Semantic chunking not fully implemented, falling back to recursive")
        return self._chunk_recursive(text, source)
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences.
        
        Args:
            text: Text to split
            
        Returns:
            List of sentences
        """
        # Simple sentence splitting using regex
        # In production, consider using more sophisticated tools like spaCy
        sentence_pattern = r'(?<=[.!?])\s+'
        sentences = re.split(sentence_pattern, text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _estimate_chunk_quality(self, chunk: Chunk) -> float:
        """Estimate the quality of a chunk.
        
        Args:
            chunk: Chunk to evaluate
            
        Returns:
            Quality score (0.0 - 1.0)
        """
        content = chunk.content
        
        # Basic quality metrics
        quality = 1.0
        
        # Penalize very short chunks
        if len(content) < 100:
            quality *= 0.5
        
        # Penalize chunks that end mid-sentence
        if not content.rstrip().endswith(('.', '!', '?', '\n')):
            quality *= 0.8
        
        # Reward chunks with complete sentences
        sentences = self._split_into_sentences(content)
        if len(sentences) >= 2:
            quality *= 1.2
        
        # Penalize chunks with too many incomplete words
        words = content.split()
        incomplete_words = sum(1 for word in words if not word.isalpha() and len(word) > 1)
        if incomplete_words / len(words) > 0.3:
            quality *= 0.7
        
        return min(quality, 1.0)
    
    def optimize_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """Optimize chunk boundaries for better quality.
        
        Args:
            chunks: Initial chunks to optimize
            
        Returns:
            Optimized chunks
        """
        if not chunks:
            return chunks
        
        optimized = []
        
        for chunk in chunks:
            quality = self._estimate_chunk_quality(chunk)
            
            # If quality is low, try to improve
            if quality < 0.7 and len(chunk.content) > 200:
                # Try to find a better break point
                content = chunk.content
                sentences = self._split_into_sentences(content)
                
                if len(sentences) > 1:
                    # Find the best split point
                    best_split = len(sentences) // 2
                    first_part = ' '.join(sentences[:best_split])
                    second_part = ' '.join(sentences[best_split:])
                    
                    if len(first_part) > 50 and len(second_part) > 50:
                        # Create two chunks
                        optimized.append(Chunk(
                            content=first_part,
                            chunk_index=chunk.chunk_index,
                            metadata={**chunk.metadata, "optimized": True},
                            start_char=chunk.start_char,
                            end_char=chunk.start_char + len(first_part),
                        ))
                        
                        optimized.append(Chunk(
                            content=second_part,
                            chunk_index=chunk.chunk_index + 0.5,  # Fractional index
                            metadata={**chunk.metadata, "optimized": True},
                            start_char=chunk.start_char + len(first_part),
                            end_char=chunk.end_char,
                        ))
                        continue
            
            # Keep original chunk
            optimized.append(chunk)
        
        # Re-index chunks
        for i, chunk in enumerate(optimized):
            chunk.chunk_index = i
        
        logger.debug(f"Optimized {len(chunks)} chunks to {len(optimized)} chunks")
        return optimized