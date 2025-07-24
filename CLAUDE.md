# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with this AgentUp RAG plugin.

## Plugin Overview

This is an AgentUp plugin that provides RAG (Retrieval-Augmented Generation) functionality with semantic search and document retrieval capabilities. It follows the A2A-specification compliant plugin architecture using the pluggy hook system.

## Plugin Structure

```
agentup-rag/
├── src/
│   └── agentup_rag/
│       ├── __init__.py
│       ├── plugin.py              # Main plugin implementation
│       ├── models.py               # Pydantic V2 data models and validation
│       ├── backends/
│       │   ├── base.py            # Abstract base classes
│       │   ├── factory.py         # Backend factory and registry
│       │   ├── embedding/         # Embedding backend implementations
│       │   └── vector_store/      # Vector store backend implementations
│       └── processing/
│           ├── chunking.py        # Text chunking strategies
│           ├── document_processor.py  # Document processing coordination
│           └── loaders.py         # Document loaders
├── tests/
│   └── test_plugin.py             # Unit tests
├── static/                        # Static assets
├── pyproject.toml                 # Package configuration with AgentUp entry point
├── README.md                      # Plugin documentation
├── RAG_SETUP_GUIDE.md            # User setup guide with API examples
└── CLAUDE.md                      # This file
```

## A2A Specification Compliance

This plugin is designed to be fully A2A-specification compliant. Always consult the A2A specification when making architectural decisions or implementing features.

## Core Plugin Architecture

### Hook System
The plugin uses pluggy hooks to integrate with AgentUp:

- `@hookimpl def register_capability()` - **Required** - Registers individual plugin capabilities
- `@hookimpl def validate_config()` - Optional - Validates plugin configuration
- `@hookimpl def execute_skill()` - **Required** - Main skill execution logic with sync signature
- `@hookimpl def get_ai_functions()` - Optional - Provides AI-callable functions

### Entry Point
The plugin is registered via entry point in `pyproject.toml`:
```toml
[project.entry-points."agentup.capabilities"]
rag = "agentup_rag.plugin:RAGPlugin"
```

## Development Guidelines

### Code Style
- Follow PEP 8 and Python best practices
- Use type hints throughout the codebase (prefer `dict` and `list` over `Dict` and `List`)
- Use `X | Y` instead of `Optional[X]` for type hints
- Use async/await for I/O operations within AI functions
- Handle errors gracefully with proper A2A error responses
- Always use `raise ... from err` in except blocks
- No f-strings without placeholders
- No emojis in code or documentation

### Plugin Implementation Patterns

#### 1. Capability Registration
```python
@hookimpl
def register_capability(self) -> List[CapabilityInfo]:
    return [
        CapabilityInfo(
            capability_id="index_document",
            name="Index Document",
            description="Add documents to the RAG vector index",
            version="0.1.0",
            capability_type=CapabilityType.AI_FUNCTION,
            tags=["rag", "indexing", "document"],
            config_schema={
                # JSON schema for configuration validation
            }
        ),
        # ... other capabilities
    ]
```

#### 2. Sync Execute Skill (AgentUp Pattern)
```python
@hookimpl
def execute_skill(self, context: CapabilityContext) -> CapabilityResult:
    """
    Main entry point - must be sync per AgentUp architecture.
    Uses asyncio.run() for async operations.
    """
    try:
        # Initialize backends if needed
        if not self.embedding_backend or not self.vector_backend:
            import asyncio
            asyncio.run(self._initialize_backends(context.config))

        # Route to appropriate handler
        capability_id = context.task.capability_id
        if capability_id == "index_document":
            return asyncio.run(self._handle_index_document(context))
        # ... other capability handlers
        
    except Exception as e:
        return CapabilityResult(
            success=False,
            content=f"Error executing {capability_id}: {str(e)}",
            error=str(e)
        )
```

#### 3. AI Function Support
```python
@hookimpl
def get_ai_functions(self) -> List[AIFunction]:
    return [
        AIFunction(
            name="index_document",
            description="Add a document to the RAG vector index",
            parameters={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The document content to index"
                    },
                    "source": {
                        "type": "string",
                        "description": "Source identifier for the document"
                    }
                },
                "required": ["content", "source"]
            },
            handler=self._ai_index_document
        )
    ]

async def _ai_index_document(self, content: str, source: str, **kwargs) -> str:
    """AI function handler - can be async"""
    # Implementation here
```

### Parameter Extraction Pattern
```python
def _extract_parameters(self, context: CapabilityContext) -> dict:
    """Extract parameters from task metadata or context metadata"""
    if hasattr(context.task, 'metadata') and context.task.metadata:
        return context.task.metadata
    else:
        return context.metadata.get("parameters", {})
```

### Error Handling
- Always return CapabilityResult objects from execute_skill
- Use success=False for errors
- Include descriptive error messages
- Log errors appropriately for debugging
- Use structured logging with context

### Testing
- Write comprehensive tests for all plugin functionality
- Test both success and error cases
- Mock external dependencies appropriately
- Use pytest and async test patterns
- Follow the updated test patterns in `tests/test_plugin.py`

### Configuration
- Define configuration schema in register_capability()
- Validate configuration in validate_config() hook
- Use environment variables for sensitive data: `${VAR_NAME:default}`
- Provide sensible defaults
- Plugin configuration is automatically passed from agentup.yml

## Development Workflow

### Local Development
1. Install in development mode: `pip install -e .`
2. Create test agent: `agentup agent create test-agent --template minimal`
3. Configure plugin in agent's `agentup.yml`:
   ```yaml
   capabilities:
     - capability_id: index_document
       required_scopes: ["rag:access"]
     - capability_id: semantic_search
       required_scopes: ["rag:access"]
   ```
4. Test with: `agentup agent serve`

### Testing
```bash
# Run tests
pytest tests/ -v

# Run with coverage
pytest --cov=agentup_rag tests/

# Check plugin loading
agentup plugin list

# Validate plugin
agentup plugin validate rag
```

### Code Quality
```bash
# Format code with ruff
ruff format src/ tests/

# Lint with ruff
ruff check src/ tests/

# Type checking
mypy src/
```

## Plugin Capabilities

### Available Capabilities
- `index_document` - Add documents to the vector index
- `semantic_search` - Search through indexed documents  
- `ask_documents` - RAG-powered question answering
- `list_collections` - List available document collections

### Backend Architecture
- **Embedding Backends**: OpenAI, local models (sentence-transformers)
- **Vector Store Backends**: In-memory (FAISS), Chroma, Pinecone, Weaviate
- **Document Processing**: Various chunking strategies (fixed, recursive, semantic)

### Middleware Support
The plugin integrates with AgentUp's auto-application middleware:
- Rate limiting
- Caching
- Retry logic
- Logging
- Validation

## Best Practices

### Performance
- Use async/await for I/O operations in AI function handlers
- Implement caching for expensive operations (embeddings)
- Use connection pooling for external APIs
- Batch processing for multiple documents
- Minimize blocking operations

### Security
- Validate all inputs with Pydantic models
- Sanitize outputs
- Use secure authentication methods (API keys via environment variables)
- Never log sensitive data
- Follow scope-based access control patterns

### Maintainability
- Follow single responsibility principle
- Keep functions small and focused
- Use descriptive variable names
- Add docstrings to all public methods
- Separate concerns between backends and main plugin logic

## Common Patterns

### Async Initialization
```python
async def _initialize_backends(self, config: Dict[str, Any]) -> None:
    """Initialize embedding and vector backends asynchronously"""
    if not self.embedding_backend:
        self.embedding_backend = create_embedding_backend(
            config.get("embedding_backend", "openai"),
            config.get("embedding_config", {})
        )
    
    if not self.vector_backend:
        self.vector_backend = create_vector_store_backend(
            config.get("vector_backend", "memory"),
            config.get("vector_config", {})
        )
        await self.vector_backend.initialize()
```

### Document Processing Pipeline
```python
async def _process_document(self, content: str, metadata: dict = None) -> List[DocumentChunk]:
    """Process document through chunking pipeline"""
    chunks = self.text_processor.chunk_text(content)
    embeddings = await self.embedding_backend.embed_texts([chunk.text for chunk in chunks])
    
    for chunk, embedding in zip(chunks, embeddings):
        chunk.embedding = embedding
        chunk.metadata.update(metadata or {})
    
    return chunks
```

## Debugging Tips

### Common Issues
- **Plugin not loading**: Check entry point in pyproject.toml
- **Configuration errors**: Verify agentup.yml capability configuration
- **Backend initialization failures**: Check API keys and network connectivity
- **Parameter extraction issues**: Verify task.metadata vs context.metadata usage
- **Async/sync mismatch**: Ensure execute_skill is sync, AI functions can be async

### Logging
```python
import logging
logger = logging.getLogger(__name__)

async def _handle_search(self, context: CapabilityContext) -> CapabilityResult:
    logger.info("Processing search request", extra={
        "capability_id": context.task.capability_id,
        "query_length": len(query)
    })
```

### Testing Patterns
```python
def test_capability_registration():
    plugin = RAGPlugin()
    capabilities = plugin.register_capability()
    assert isinstance(capabilities, list)
    assert len(capabilities) == 4
    
    index_cap = next(cap for cap in capabilities if cap.capability_id == "index_document")
    assert index_cap.name == "Index Document"
```

## Important Architecture Notes

### AgentUp Integration
- Plugin configuration flows from agentup.yml → AgentUp framework → CapabilityContext.config
- Individual capabilities are registered separately (not as a single plugin)
- Scope-based security is handled automatically by the framework
- Mixed sync/async pattern: plugin hooks are sync, AI functions can be async

### Pydantic V2 Migration
- All models use Pydantic V2 syntax with `@field_validator` and `@model_validator`
- Use `@classmethod` with field validators
- Proper error handling and validation throughout

### A2A Protocol
- All responses must be A2A-compliant
- Use proper message/send method for API communication
- Follow JSON-RPC 2.0 format for all interactions
- Support all A2A artifact types

## Resources

- [AgentUp Documentation](https://docs.agentup.dev)
- [A2A Specification](https://a2a.dev)
- [Plugin Development Guide](https://docs.agentup.dev/plugins/development)
- [RAG Setup Guide](RAG_SETUP_GUIDE.md) - User-facing setup instructions
- [README](README.md) - Comprehensive plugin documentation

Remember: This plugin is part of the AgentUp ecosystem. Always consider how it integrates with other plugins and follows A2A standards for maximum compatibility and usefulness.