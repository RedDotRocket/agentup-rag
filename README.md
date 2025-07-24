# AgentUp RAG Plugin

<p align="center">
  <img src="static/logo.png" alt="Compie Logo" width="400"/>
</p>

Note this plugin is currently in development and not extensively tested. It is intended for early adopters and contributors. If you encounter issues, please open an issue on GitHub, or even better, submit a pull request with a fix! If anyone is interested in ownership of this plugin, and you have proof of  previously
being a good custodian of a project, please reach out.

## Features

- **Multiple Embedding Backends**: OpenAI, local models (sentence-transformers)
- **Multiple Vector Store Backends**: In-memory (FAISS), Chroma, Pinecone, Weaviate
- **Document Processing Pipeline**: Various chunking strategies (fixed, recursive, semantic)
- **AI Function Integration**: LLM-callable functions for natural interaction
- **Production Features**: Caching, rate limiting, monitoring, security
- **AgentUp Native**: Deep integration with AgentUp's plugin system and middleware

## Installation

### From Source

```bash
# Clone or navigate to the plugin directory
cd agentup-rag

# Install in development mode
pip install -e .

# For GPU acceleration (requires Python 3.12 or earlier)
pip install -e ".[gpu]"

# Verify installation
agentup plugin list
```

### From PyPI (when published)

```bash
# Standard installation
pip install agentup-rag

# With GPU support (requires Python 3.12 or earlier)
pip install "agentup-rag[gpu]"
```

**Note**: GPU acceleration via FAISS-GPU is only available for Python 3.9-3.12. Python 3.13+ users will automatically use CPU-only FAISS.

## Quick Start

### 1. Basic Configuration

Create an agent configuration with the RAG plugin:

```yaml
# agent_config.yaml
agent:
  name: "RAG Agent"
  description: "Agent with RAG capabilities"

plugins:
  - plugin_id: rag
    visibility: "extended"  # Hide from public discovery (optional)
    required_scopes: ["data:read", "ai:execute"]  # Minimum required scopes
    config:
      # Embedding backend configuration
      embedding_backend: "openai"
      embedding_config:
        model: "text-embedding-3-small"
        api_key: "${OPENAI_API_KEY}"
        batch_size: 100

      # Vector store backend configuration
      vector_backend: "memory"
      vector_config:
        similarity_metric: "cosine"
        index_type: "flat"
        persist_path: "./vector_index.pkl"

      # Document processing configuration
      chunking:
        strategy: "recursive"
        chunk_size: 1000
        chunk_overlap: 200

      # Search configuration
      search:
        default_k: 5
        max_k: 50
        similarity_threshold: 0.0
```

### 2. Plugin Visibility and Access Control

Configure plugin visibility and access control based on your deployment needs:

```yaml
plugins:
  - plugin_id: rag
    # Visibility control
    visibility: "extended"  # "public" or "extended"
    
    # Minimum required scopes for plugin access
    required_scopes: ["data:read", "ai:execute"]
    
    config:
      # ... plugin configuration
```

**Visibility Options:**
- `"public"` - Plugin appears in public Agent Card (default)
- `"extended"` - Plugin only appears in authenticated Extended Agent Card

**Scope Configuration Examples:**

```yaml
# Read-only RAG access
plugins:
  - plugin_id: rag
    required_scopes: ["data:read"]
    
# Full RAG capabilities
plugins:
  - plugin_id: rag
    required_scopes: ["data:read", "data:write", "data:process", "ai:execute"]
    
# Administrative access
plugins:
  - plugin_id: rag
    required_scopes: ["data:admin", "ai:execute", "api:external"]
```

### 3. Environment Variables

Set required environment variables:

```bash
# For OpenAI embeddings
export OPENAI_API_KEY="sk-your-key-here"

# For Pinecone (if using)
export PINECONE_API_KEY="your-pinecone-key"
export PINECONE_ENVIRONMENT="us-west1-gcp"
```

### 3. Start Your Agent

```bash
agentup agent serve
```

## Usage

The RAG plugin provides several AI functions that can be called by LLMs or used directly:

### Index Documents

```python
# Add a document to the search index
result = await agent.call_function("index_document", {
    "content": "Your document content here...",
    "source": "document.txt",
    "collection": "my_documents",
    "metadata": {"author": "John Doe", "category": "research"}
})
```

### Semantic Search

```python
# Search for relevant documents
result = await agent.call_function("semantic_search", {
    "query": "machine learning algorithms",
    "collection": "my_documents",
    "k": 5,
    "similarity_threshold": 0.3
})
```

### Ask Questions (RAG)

```python
# Get answers based on indexed documents
result = await agent.call_function("ask_documents", {
    "question": "What are the main benefits of deep learning?",
    "collection": "my_documents",
    "max_context_length": 4000,
    "include_sources": True
})
```

### List Collections

```python
# See available document collections
result = await agent.call_function("list_collections", {})
```

## Configuration Options

### Embedding Backends

#### OpenAI

```yaml
embedding_backend: "openai"
embedding_config:
  model: "text-embedding-3-small"  # or "text-embedding-3-large"
  api_key: "${OPENAI_API_KEY}"
  organization: "${OPENAI_ORG_ID}"  # optional
  batch_size: 100
  rate_limit: 60  # requests per minute
  max_retries: 3
  timeout: 30
```

#### Local Models

```yaml
embedding_backend: "local"
embedding_config:
  model: "all-MiniLM-L6-v2"  # or other sentence-transformers models
  device: "cpu"  # or "cuda", "mps"
  batch_size: 32
  cache_dir: "./model_cache"
  normalize_embeddings: true
```

### Vector Store Backends

#### Memory (FAISS)

```yaml
vector_backend: "memory"
vector_config:
  similarity_metric: "cosine"  # "euclidean", "dot_product"
  index_type: "flat"  # "ivf", "hnsw"
  persist_path: "./vector_index.pkl"
  auto_save: true
  save_interval: 300  # seconds
```

#### Chroma

```yaml
vector_backend: "chroma"
vector_config:
  persist_directory: "./chroma_db"
  collection_name: "documents"
  host: "localhost"  # for remote Chroma
  port: 8000
  api_key: "${CHROMA_API_KEY}"  # if auth enabled
```

#### Pinecone

```yaml
vector_backend: "pinecone"
vector_config:
  api_key: "${PINECONE_API_KEY}"
  environment: "${PINECONE_ENVIRONMENT}"
  index_name: "my-rag-index"
  dimension: 1536  # must match embedding model
  metric: "cosine"
```

### Document Processing

```yaml
chunking:
  strategy: "recursive"  # "fixed", "semantic"
  chunk_size: 1000
  chunk_overlap: 200
  separators: ["\n\n", "\n", ".", "!", "?"]
```

## Development

### Running Tests

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run with coverage
pytest --cov=agentup_rag tests/

# Run specific test file
pytest tests/test_plugin.py -v
```

### Code Quality

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Type checking
mypy src/

# Linting
flake8 src/ tests/
```

### Adding New Backends

To add a new embedding or vector store backend:

1. Create the backend implementation inheriting from the appropriate base class
2. Register it in the factory with `register_embedding_backend()` or `register_vector_store_backend()`
3. Add configuration models in `models.py`
4. Add tests for the new backend

## Architecture

The plugin follows a clean architecture with clear separation of concerns:

```
agentup_rag/
├── plugin.py              # Main plugin class with AgentUp integration
├── models.py               # Pydantic data models and configuration
├── backends/
│   ├── base.py            # Abstract base classes
│   ├── factory.py         # Backend factory and registry
│   ├── embedding/         # Embedding backend implementations
│   └── vector_store/      # Vector store backend implementations
└── processing/
    ├── chunking.py        # Text chunking strategies
    ├── document_processor.py  # Document processing coordination
    └── loaders.py         # Document loaders (placeholder)
```

## Performance Considerations

- **Embedding Caching**: Embeddings are cached to avoid recomputation
- **Batch Processing**: Optimized batch processing for multiple documents
- **Connection Pooling**: HTTP clients use connection pooling for efficiency
- **Rate Limiting**: Built-in rate limiting to respect API limits
- **Memory Management**: Efficient memory usage for large document collections

## Security & Privacy

The RAG plugin implements comprehensive security features:

### Authentication & Authorization

- **Scope-based Access Control**: Fine-grained permissions for different operations
- **Required Scopes**: 
  - `data:read` - Read access to documents and search
  - `data:write` - Write access to index documents
  - `data:process` - Process and transform documents
  - `ai:execute` - Execute AI-powered functions
  - `data:admin` - Administrative operations
  - `data:sensitive` - Access to sensitive content
  - `api:external` - Call external APIs (for OpenAI embeddings)

### Capability IDs and Scopes

The RAG plugin provides the following capabilities with their associated scope requirements:

| Capability ID | Recommended Scopes | Description |
|---------------|-------------------|-------------|
| `index_document` | `data:process`, `data:write` | Add documents to the vector index |
| `semantic_search` | `data:read` | Search through indexed documents |
| `ask_documents` | `data:read`, `ai:execute` | RAG-powered question answering |
| `list_collections` | `data:read` | List available document collections |

**Note**: The current AgentUp framework automatically handles scope enforcement. The scopes listed above are recommendations for production deployments.

### Security Features

- **Content-based Access Control**: Automatic detection of sensitive content markers
- **Audit Logging**: Comprehensive logging for compliance and monitoring
- **Input Validation**: Strict validation of all inputs and parameters
- **Error Handling**: Secure error messages that don't leak sensitive information

## Monitoring

The plugin includes built-in monitoring capabilities:

- Performance metrics (search latency, indexing time)
- Usage statistics (queries, document counts)
- Health checks for all backends
- Structured logging with context

## Troubleshooting

### Common Issues

1. **"Backend not initialized" error**
   - Ensure all required configuration is provided
   - Check that API keys are set correctly
   - Verify network connectivity for remote services

2. **High memory usage**
   - Consider using remote vector stores for large datasets
   - Adjust chunk sizes and batch sizes
   - Enable vector quantization (future feature)

3. **Slow search performance**
   - Use approximate search algorithms (IVF, HNSW)
   - Enable result caching
   - Consider using faster embedding models

### Debug Mode

Enable debug logging to troubleshoot issues:

```python
import logging
logging.getLogger("agentup_rag").setLevel(logging.DEBUG)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

This project is licensed under the Apache License 2.0. See the LICENSE file for details.

## Support

For support and questions:

- Open an issue on GitHub
- Check the documentation
- Join the AgentUp community discussions

## Roadmap

See the [technical design document](design-docs/RAG_PLUGIN_TECHNICAL_DESIGN.md) for planned features and implementation phases.

## Changelog

### Version 0.1.0

- Initial release
- OpenAI and memory backends
- Basic document processing
- AI function integration
- AgentUp middleware support