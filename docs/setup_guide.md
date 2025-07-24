# AgentUp RAG Plugin Setup Guide

This guide walks you through setting up an AgentUp agent with RAG (Retrieval-Augmented Generation) capabilities using the AgentUp RAG plugin.

The RAG plugin enables your agents to perform semantic search, document retrieval, and question-answering by leveraging vector stores and embedding models. It supports various vector stores like Chroma, Pinecone, Weaviate, and Memory Store, and can use OpenAI embeddings or local models.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Vector Store Setup](#vector-store-setup)
4. [Embedding Provider Setup](#embedding-provider-setup)
5. [Agent Configuration](#agent-configuration)
6. [Testing Your RAG Agent](#testing-your-rag-agent)
7. [Example Usage](#example-usage)
8. [Advanced Usage](#advanced-usage)
9. [Troubleshooting](#troubleshooting)

## Prerequisites

- Python 3.11 or higher (3.13 is not supported by faiss-gpu yet!)
- AgentUp framework installed
- API keys for your chosen embedding provider (OpenAI recommended), or local embedding model (e.g., `all-MiniLM-L6-v2`) set up
- Access to a vector store (Chroma, Pinecone, Weaviate, or Memory Store)

## Installation


### Pull from the AgentUp Plugin Registry

```bash
pip install agentup-rag --index-url https://api.agentup.dev/simple
```

### Install the RAG Plugin (local development)

```bash
# Navigate to your AgentUp workspace
cd /path/to/agentup-workspace

# Activate your virtual environment
source .venv/bin/activate

# Install the RAG plugin
pip install -e plugins/agentup-rag/
```

### 2. Verify Installation

```bash
# Check if the plugin is available
agentup plugin list
```

You should see the RAG plugin listed with status "loaded".

## Vector Store Setup

Choose one of the following vector store options:

### Option A: Memory Store (Development/Testing)

No additional setup required. This stores vectors in memory and optionally persists to disk.

**Pros:** Easy setup, no external dependencies
**Cons:** Limited scalability, data lost on restart (unless persisted)

### Option B: Chroma (Local Production)

```bash
# Install Chroma
pip install chromadb

# Optional: Run Chroma server (for remote access)
# chroma run --host localhost --port 8000
```

**Pros:** Persistent storage, good performance, local control
**Cons:** Single machine deployment

### Option C: Pinecone (Cloud Production)

1. Sign up at [Pinecone](https://www.pinecone.io/)
2. Create an API key
3. Note your environment (e.g., "us-west1-gcp-free")

```bash
# Install Pinecone client
pip install pinecone
```

**Pros:** Managed service, high availability, excellent scalability
**Cons:** Requires cloud service, has costs

### Option D: Weaviate (Enterprise)

```bash
# Install Weaviate client
pip install weaviate-client

# Option 1: Run local Weaviate with Docker
docker run -p 8080:8080 \
  -e QUERY_DEFAULTS_LIMIT=25 \
  -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true \
  -e PERSISTENCE_DATA_PATH='/var/lib/weaviate' \
  semitechnologies/weaviate:latest

# Option 2: Use Weaviate Cloud Services (WCS)
# Sign up at https://console.weaviate.cloud/
```

**Pros:** Advanced features, graph capabilities, multi-modal support
**Cons:** More complex setup, learning curve

## Embedding Provider Setup

### Option A: OpenAI Embeddings (Recommended)

1. Get an API key from [OpenAI](https://platform.openai.com/api-keys)
2. Set environment variable:

```bash
export OPENAI_API_KEY="sk-your-api-key-here"
```

**Pros:** High quality embeddings, reliable service
**Cons:** Requires API calls, has costs

### Option B: Local Models (Free)

No API key required, but requires more computational resources.

**Pros:** No API costs, full data privacy
**Cons:** Slower, requires more RAM/CPU

## Agent Configuration

### 1. Create a New Agent

```bash
# Create a new agent
agentup agent create my-rag-agent
cd my-rag-agent
```

### 2. Configure the Agent

Edit the `agentup.yml` file with one of the following configurations:

#### Basic Configuration with Memory Store + OpenAI

```yaml
# Agent Information
agent:
  name: RAG Agent
  description: AI agent with semantic search and document retrieval capabilities
  version: 1.0.0

# Plugins Configuration
plugins:
  - plugin_id: rag
    name: "RAG Search & Retrieval"
    description: "Semantic search and document retrieval"

    # Explicit capability configuration for security
    capabilities:
      - capability_id: index_document
        required_scopes: ["rag:access"]
      - capability_id: semantic_search
        required_scopes: ["rag:access"]
      - capability_id: ask_documents
        required_scopes: ["rag:access"]
      - capability_id: list_collections
        required_scopes: ["rag:access"]

    config:
      # Embedding backend configuration
      embedding_backend: "openai"
      embedding_config:
        model: "text-embedding-3-small"
        api_key: "${OPENAI_API_KEY}"
        batch_size: 100
        rate_limit: 1000

      # Vector store backend configuration
      vector_backend: "memory"
      vector_config:
        similarity_metric: "cosine"
        persist_path: "./vector_index.pkl"
        auto_save: true
        save_interval: 300

      # Document processing configuration
      chunking:
        strategy: "recursive"
        chunk_size: 1000
        chunk_overlap: 200
        separators: ["\n\n", "\n", ".", "!", "?"]

      # Search configuration
      search:
        default_k: 5
        max_k: 100
        similarity_threshold: 0.0
        include_metadata: true

      # RAG configuration
      rag:
        max_context_length: 4000
        include_sources: true

      # Collection management
      collections:
        default_collection: "documents"
        auto_create: true
        max_collections: 10

# AI Provider Configuration
ai_provider:
  provider: "openai"
  model: "gpt-4o-mini"
  api_key: "${OPENAI_API_KEY}"
  temperature: 0.1
  max_tokens: 2000

# Security configuration (required for scope-based access control)
security:
  enabled: true
  type: "api_key"
  auth:
    api_key:
      header_name: "X-API-Key"
      location: "header"
      keys:
        - key: "your-secure-api-key-here"
          scopes: ["api:read", "api:write", "rag:access"]

  # Scope hierarchy
  scope_hierarchy:
    admin: ["*"]
    api:write: ["api:read"]
    api:read: []
    rag:access: []

# Routing configuration
routing:
  default_mode: ai

# Logging configuration
logging:
  enabled: true
  level: "INFO"
  format: "text"
  console:
    enabled: true
    colors: true
```

#### Production Configuration with Pinecone

```yaml
# Agent Information
agent:
  name: Production RAG Agent
  description: Production RAG agent with Pinecone vector database
  version: 1.0.0

# Plugins Configuration
plugins:
  - plugin_id: rag
    name: "RAG Search & Retrieval"
    description: "Production RAG with Pinecone"

    capabilities:
      - capability_id: index_document
        required_scopes: ["rag:access"]
      - capability_id: semantic_search
        required_scopes: ["rag:access"]
      - capability_id: ask_documents
        required_scopes: ["rag:access"]
      - capability_id: list_collections
        required_scopes: ["rag:access"]

    config:
      # Embedding backend configuration
      embedding_backend: "openai"
      embedding_config:
        model: "text-embedding-3-small"
        api_key: "${OPENAI_API_KEY}"
        batch_size: 100
        rate_limit: 1000

      # Vector store backend configuration
      vector_backend: "pinecone"
      vector_config:
        api_key: "${PINECONE_API_KEY}"
        environment: "${PINECONE_ENVIRONMENT}"
        index_name: "my-rag-index"
        dimension: 1536  # Must match embedding model
        metric: "cosine"
        deployment_type: "serverless"  # or "pod"
        cloud: "aws"
        region: "us-east-1"

      # Document processing configuration
      chunking:
        strategy: "recursive"
        chunk_size: 1000
        chunk_overlap: 200

      # Search configuration
      search:
        default_k: 5
        max_k: 50
        similarity_threshold: 0.1

      # RAG configuration
      rag:
        max_context_length: 8000
        include_sources: true

# AI Provider Configuration
ai_provider:
  provider: "openai"
  model: "gpt-4"
  api_key: "${OPENAI_API_KEY}"
  temperature: 0.1
  max_tokens: 2000

# Security configuration
security:
  enabled: true
  type: "api_key"
  auth:
    api_key:
      header_name: "X-API-Key"
      location: "header"
      keys:
        - key: "your-secure-api-key-here"
          scopes: ["api:read", "api:write", "rag:access"]

  scope_hierarchy:
    admin: ["*"]
    api:write: ["api:read"]
    api:read: []
    rag:access: []

# Environment variables needed:
# OPENAI_API_KEY=sk-your-openai-key
# PINECONE_API_KEY=your-pinecone-key
# PINECONE_ENVIRONMENT=your-pinecone-environment
```

#### Local Setup with Chroma

```yaml
plugins:
  - plugin_id: rag
    capabilities:
      - capability_id: index_document
        required_scopes: ["rag:access"]
      - capability_id: semantic_search
        required_scopes: ["rag:access"]
      - capability_id: ask_documents
        required_scopes: ["rag:access"]
      - capability_id: list_collections
        required_scopes: ["rag:access"]

    config:
      embedding_backend: "local"
      embedding_config:
        model: "all-MiniLM-L6-v2"
        device: "cpu"  # or "cuda" if you have GPU
        batch_size: 32
        normalize_embeddings: true

      vector_backend: "chroma"
      vector_config:
        persist_directory: "./chroma_db"
        collection_name: "documents"
        distance_function: "cosine"
        # Optional: for remote Chroma server
        # host: "localhost"
        # port: 8000
```

### 3. Set Environment Variables

Create a `.env` file in your agent directory:

```bash
# .env file
OPENAI_API_KEY=sk-your-openai-api-key-here

# If using Pinecone
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_ENVIRONMENT=us-west1-gcp-free

# If using Weaviate Cloud
WEAVIATE_API_KEY=your-weaviate-api-key
```

## Testing Your RAG Agent

### 1. Start the Agent

```bash
# Navigate to your agent directory
cd my-rag-agent

# Start the agent
agentup agent serve
```


## Example Usage

### 1. Index a Document

```bash
curl -X POST http://localhost:8000/ \
  -H "X-API-Key: your-secure-api-key-here" \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "message/send",
    "params": {
      "message": {
        "role": "user",
        "parts": [{
          "kind": "text",
          "text": "Index this document: AgentUp is a comprehensive framework for creating awesome AI agents. It provides tools for building, deploying, and managing AI agents with features like plugin systems, state management. Source: agentup_intro.txt, Collection: knowledge_base"
        }],
        "messageId": "msg-001",
        "kind": "message"
      }
    },
    "id": "req-001"
  }'
```

### 2. Search for Similar Content

```bash
curl -X POST http://localhost:8000/ \
  -H "X-API-Key: your-secure-api-key-here" \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "message/send",
    "params": {
      "message": {
        "role": "user",
        "parts": [{
          "kind": "text",
          "text": "Search for: What is AgentUp framework?"
        }],
        "messageId": "msg-002",
        "kind": "message"
      }
    },
    "id": "req-002"
  }'
```

### 3. Ask Questions (RAG)

```bash
curl -X POST http://localhost:8000/ \
  -H "X-API-Key: your-secure-api-key-here" \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "message/send",
    "params": {
      "message": {
        "role": "user",
        "parts": [{
          "kind": "text",
          "text": "What are the key features of AgentUp?"
        }],
        "messageId": "msg-003",
        "kind": "message"
      }
    },
    "id": "req-003"
  }'
```

### 4. Using with a Chat Interface

If you have a chat interface connected to your agent, you can simply ask questions like:

- "What is AgentUp?"
- "Tell me about AI agent frameworks"
- "How does AgentUp handle plugins?"

The agent will automatically use RAG to search your indexed documents and provide informed answers.

### 5. Direct Function Call Format

For more explicit control, you can use structured requests:

```bash
curl -X POST http://localhost:8000/ \
  -H "X-API-Key: your-secure-api-key-here" \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "message/send",
    "params": {
      "message": {
        "role": "user",
        "parts": [{
          "kind": "text",
          "text": "Please call the index_document function with these parameters: content=\"AgentUp is a comprehensive framework for creating production-ready AI agents.\", source=\"agentup_intro.txt\", collection=\"knowledge_base\""
        }],
        "messageId": "msg-004",
        "kind": "message"
      }
    },
    "id": "req-004"
  }'
```

## Advanced Usage

### Bulk Document Indexing

Create a script to index multiple documents:

```python
import requests
import json
import os

def index_document(content, source, collection="documents", api_key="your-secure-api-key-here"):
    """Index a single document"""
    payload = {
        "jsonrpc": "2.0",
        "method": "message/send",
        "params": {
            "message": {
                "role": "user",
                "parts": [{
                    "kind": "text",
                    "text": f"Index this document: {content}\nSource: {source}\nCollection: {collection}"
                }],
                "messageId": f"msg-{source}",
                "contextId": "bulk-index-001",
                "kind": "message"
            }
        },
        "id": 1
    }

    response = requests.post(
        "http://localhost:8000/",
        headers={
            "Content-Type": "application/json",
            "X-API-Key": api_key
        },
        data=json.dumps(payload)
    )

    return response.json()

def index_directory(directory_path, collection="documents"):
    """Index all text files in a directory"""
    for filename in os.listdir(directory_path):
        if filename.endswith(('.txt', '.md')):
            file_path = os.path.join(directory_path, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            result = index_document(content, filename, collection)
            print(f"Indexed {filename}: {result}")

# Usage
index_directory("./documents", "my_knowledge_base")
```

### Multiple Collections

You can organize documents into different collections:

```python
# Index different types of documents
index_document(content, "tech_doc.txt", "technical")
index_document(content, "policy.txt", "policies")
index_document(content, "faq.txt", "support")

# Search within specific collections by mentioning them in queries
# "Search for technical information in the technical collection"
```

## Troubleshooting

### Common Issues

#### 1. Plugin Not Loading

**Error:** `Plugin rag did not return capability info`

**Solution:** Ensure the plugin is properly installed:
```bash
pip install -e plugins/agentup-rag/
agentup plugin list
```

#### 2. Missing Capability Configuration

**Error:** `Plugin 'rag' uses legacy format - explicit capability configuration required`

**Solution:** Update your agent configuration to use individual capabilities:
```yaml
plugins:
  - plugin_id: rag
    capabilities:
      - capability_id: index_document
        required_scopes: ["rag:access"]
      - capability_id: semantic_search
        required_scopes: ["rag:access"]
      - capability_id: ask_documents
        required_scopes: ["rag:access"]
      - capability_id: list_collections
        required_scopes: ["rag:access"]
    config:
      # ... rest of your config
```

Also ensure your security configuration includes the `rag:access` scope:
```yaml
security:
  scope_hierarchy:
    rag:access: []

  auth:
    api_key:
      keys:
        - key: "your-key"
          scopes: ["api:read", "api:write", "rag:access"]
```

#### 3. Vector Store Connection Issues

**Error:** `Failed to initialize [backend] backend`

**Solutions:**
- Check API keys are set correctly
- Verify network connectivity
- Check service status (for cloud providers)
- Review configuration parameters

#### 4. Embedding Generation Failures

**Error:** `Failed to generate embeddings`

**Solutions:**
- Check OpenAI API key and billing
- Verify rate limits aren't exceeded
- For local models, ensure sufficient RAM

#### 5. Search Returns No Results

**Possible causes:**
- No documents indexed yet
- Similarity threshold too high
- Wrong collection name
- Embedding dimension mismatch

**Solutions:**
```bash
# Check collections
curl -X POST http://localhost:8000/ \
  -H "X-API-Key: your-secure-api-key-here" \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "message/send",
    "params": {
      "message": {
        "role": "user",
        "parts": [{
          "kind": "text",
          "text": "List all collections"
        }],
        "messageId": "msg-debug-001",
        "contextId": "debug-context",
        "kind": "message"
      }
    },
    "id": "req-debug-001"
  }'
```

### Debug Mode

Enable debug logging in your agent config:

```yaml
logging:
  enabled: true
  level: "DEBUG"  # Changed from INFO
  format: "text"
  console:
    enabled: true
    colors: true
```

### Performance Tuning

#### For Large Document Collections

```yaml
config:
  # Increase batch sizes
  embedding_config:
    batch_size: 200

  # Optimize chunking
  chunking:
    chunk_size: 800  # Smaller chunks for better precision
    chunk_overlap: 150

  # Tune search parameters
  search:
    default_k: 10
    similarity_threshold: 0.2  # Higher threshold for quality
```

#### For Memory-Constrained Environments

```yaml
config:
  # Use smaller embedding models
  embedding_backend: "local"
  embedding_config:
    model: "all-MiniLM-L6-v2"  # 384 dimensions vs 1536
    batch_size: 16  # Smaller batches

  # Smaller chunks
  chunking:
    chunk_size: 500
    chunk_overlap: 100
```

## Next Steps

1. **Production Deployment:** Use Pinecone or Weaviate for production workloads
2. **Security:** Implement authentication and access controls
3. **Monitoring:** Set up metrics and alerting for your RAG system
4. **Optimization:** Fine-tune chunk sizes and search parameters for your use case
5. **Multi-modal:** Explore Weaviate for image and video content

## Resources

- [AgentUp Documentation](https://docs.agentup.dev/)
- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)
- [Pinecone Documentation](https://docs.pinecone.io/)
- [Chroma Documentation](https://docs.trychroma.com/)
- [Weaviate Documentation](https://weaviate.io/developers/weaviate)

## Support

For issues with the RAG plugin:
1. Check the troubleshooting section above
2. Review AgentUp logs for detailed error messages
3. Verify all dependencies are correctly installed
4. Test with minimal configuration first, then add complexity

The RAG plugin provides powerful semantic search and question-answering capabilities to your AgentUp agents. Start with a simple setup and gradually move to production-ready configurations as your needs grow.