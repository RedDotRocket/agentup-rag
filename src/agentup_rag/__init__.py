"""
AgentUp RAG Plugin

A comprehensive Retrieval-Augmented Generation plugin for AgentUp with semantic search 
and document retrieval capabilities.

This plugin provides:
- Multiple embedding backends (OpenAI, local models)
- Multiple vector store backends (memory, Chroma, Pinecone, Weaviate)
- Document processing pipeline with various chunking strategies
- AI function integration for LLM-callable capabilities
- Production-ready features like caching, monitoring, and security
"""

__version__ = "0.1.0"
__author__ = "Luke Hinds"
__email__ = "luke@rdrocket.com"

from .plugin import RAGPlugin

__all__ = ["RAGPlugin"]