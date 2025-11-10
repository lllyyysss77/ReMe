---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Vector Store Configuration Guide

This guide covers how to configure vector store backends in ReMe using the `default.yaml` configuration file.

## 📋 Overview

ReMe provides multiple vector store backends for different use cases:

- **LocalVectorStore** (`backend=local`) - 📁 Simple file-based storage for development and small datasets
- **ChromaVectorStore** (`backend=chroma`) - 🔮 Embedded vector database for moderate scale
- **EsVectorStore** (`backend=elasticsearch`) - 🔍 Elasticsearch-based storage for production and large scale
- **QdrantVectorStore** (`backend=qdrant`) - 🎯 High-performance vector database with advanced filtering
- **MemoryVectorStore** (`backend=memory`) - ⚡ In-memory storage for ultra-fast access and testing

All vector stores implement the `BaseVectorStore` interface, providing a consistent API across implementations.

## 📊 Comparison Table

| Feature              | LocalVectorStore | ChromaVectorStore | EsVectorStore | QdrantVectorStore | MemoryVectorStore |
|----------------------|------------------|-------------------|---------------|-------------------|-------------------|
| **Storage**          | File (JSONL)     | Embedded DB       | Elasticsearch | Qdrant Server     | In-Memory         |
| **Performance**      | Medium           | Good              | Excellent     | Excellent         | Ultra-Fast        |
| **Scalability**      | < 10K vectors    | < 1M vectors      | > 1M vectors  | > 10M vectors     | < 1M vectors      |
| **Persistence**      | ✅ Auto           | ✅ Auto            | ✅ Auto        | ✅ Auto            | ⚠️ Manual         |
| **Setup Complexity** | 🟢 Simple        | 🟡 Medium         | 🔴 Complex    | 🟡 Medium         | 🟢 Simple         |
| **Dependencies**     | None             | ChromaDB          | Elasticsearch | Qdrant            | None              |
| **Filtering**        | ❌ Basic          | ✅ Metadata        | ✅ Advanced    | ✅ Advanced        | ❌ Basic           |
| **Concurrency**      | ❌ Limited        | ✅ Good            | ✅ Excellent   | ✅ Excellent       | ❌ Single Process  |
| **Async Support**    | ❌ No             | ❌ No              | ❌ No          | ✅ Native          | ❌ No              |
| **Best For**         | Development      | Local Apps        | Production    | Production/Cloud  | Testing           |

## ⚙️ Configuration in default.yaml

All vector stores are configured in the `vector_store` section of `reme_ai/config/default.yaml`. The configuration structure is:

```yaml
vector_store:
  default:
    backend: <backend_name>        # Required: local, chroma, elasticsearch, qdrant, or memory
    embedding_model: default        # Required: Name of the embedding model configuration
    params:                         # Optional: Backend-specific parameters
      # Backend-specific parameters go here
```

### Configuration Fields

- **`backend`** (required): The vector store backend to use. Valid values: `local`, `chroma`, `elasticsearch`, `qdrant`, `memory`
- **`embedding_model`** (required): The name of the embedding model configuration from the `embedding_model` section
- **`params`** (optional): A dictionary of backend-specific parameters that will be passed to the vector store constructor

## 📁 Vector Store Backend Configurations

### 1. LocalVectorStore (`backend=local`)

A simple file-based vector store that saves data to local JSONL files.

#### 💡 When to Use
- **Development and testing** - No external dependencies required 🛠️
- **Small datasets** - Suitable for datasets with < 10,000 vectors 📊
- **Single-user applications** - Limited concurrent access support 👤

#### ⚙️ Configuration

```yaml
vector_store:
  default:
    backend: local
    embedding_model: default
    params:
      store_dir: "./local_vector_store"  # Directory to store JSONL files (default: "./local_vector_store")
      batch_size: 1024                    # Batch size for operations (default: 1024)
```

#### Configuration Parameters

- **`store_dir`** (optional): Directory path where workspace files are stored. Default: `"./local_vector_store"`
- **`batch_size`** (optional): Batch size for bulk operations. Default: `1024`

### 2. ChromaVectorStore (`backend=chroma`)

An embedded vector database that provides persistent storage with advanced features.

#### 💡 When to Use
- **Local development** with persistence requirements 🏠
- **Medium-scale applications** (10K - 1M vectors) 📈
- **Applications requiring metadata filtering** 🔍

#### ⚙️ Configuration

```yaml
vector_store:
  default:
    backend: chroma
    embedding_model: default
    params:
      store_dir: "./chroma_vector_store"  # Directory for Chroma database (default: "./chroma_vector_store")
      batch_size: 1024                    # Batch size for operations (default: 1024)
```

#### Configuration Parameters

- **`store_dir`** (optional): Directory path where ChromaDB data is persisted. Default: `"./chroma_vector_store"`
- **`batch_size`** (optional): Batch size for bulk operations. Default: `1024`

### 3. EsVectorStore (`backend=elasticsearch`)

Production-grade vector search using Elasticsearch with advanced filtering and scaling capabilities.

#### 💡 When to Use
- **Production environments** requiring high availability 🏭
- **Large-scale applications** (1M+ vectors) 🚀
- **Complex filtering requirements** on metadata 🎯

#### 🛠️ Setup Elasticsearch

Before using EsVectorStore, set up Elasticsearch:

##### Option 1: Docker Run
```bash
# Pull the latest Elasticsearch image
docker pull docker.elastic.co/elasticsearch/elasticsearch-wolfi:9.0.0

# Run Elasticsearch container
docker run -p 9200:9200 \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  -e "xpack.license.self_generated.type=trial" \
  -e "http.host=0.0.0.0" \
  docker.elastic.co/elasticsearch/elasticsearch-wolfi:9.0.0
```

##### Environment Configuration
```bash
export FLOW_ES_HOSTS=http://localhost:9200
```

#### ⚙️ Configuration

```yaml
vector_store:
  default:
    backend: elasticsearch
    embedding_model: default
    params:
      hosts: "http://localhost:9200"     # Elasticsearch host(s) - can be string or list (default: from FLOW_ES_HOSTS env var or "http://localhost:9200")
      basic_auth: null                    # Optional: ("username", "password") tuple for authentication
      batch_size: 1024                    # Batch size for bulk operations (default: 1024)
```

#### Configuration Parameters

- **`hosts`** (optional): Elasticsearch host(s) as a string or list of strings. Defaults to the `FLOW_ES_HOSTS` environment variable or `"http://localhost:9200"` if not set
- **`basic_auth`** (optional): Tuple of `("username", "password")` for basic authentication. Default: `null` (no authentication)
- **`batch_size`** (optional): Batch size for bulk operations. Default: `1024`

### 4. QdrantVectorStore (`backend=qdrant`)

A high-performance vector database designed for production workloads with native async support and advanced filtering.

#### 💡 When to Use
- **Production environments** requiring high performance and reliability 🏭
- **Large-scale applications** (10M+ vectors) with excellent horizontal scaling 🚀
- **Applications requiring native async operations** for better concurrency ⚡
- **Complex filtering and metadata queries** on large datasets 🎯
- **Cloud-native deployments** with Qdrant Cloud support ☁️

#### 🛠️ Setup Qdrant

Before using QdrantVectorStore, set up Qdrant:

##### Option 1: Docker Run (Recommended for Development)
```bash
# Pull the latest Qdrant image
docker pull qdrant/qdrant

# Run Qdrant container
docker run -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage:z \
  qdrant/qdrant
```

##### Option 2: Qdrant Cloud
For production, you can use [Qdrant Cloud](https://cloud.qdrant.io/) for managed hosting.

##### Environment Configuration
```bash
# For local setup
export FLOW_QDRANT_HOST=localhost
export FLOW_QDRANT_PORT=6333

# For cloud setup (optional)
export FLOW_QDRANT_API_KEY=your-api-key
```

#### ⚙️ Configuration

##### Local Qdrant Instance
```yaml
vector_store:
  default:
    backend: qdrant
    embedding_model: default
    params:
      host: "localhost"                   # Qdrant host (default: from FLOW_QDRANT_HOST env var or "localhost")
      port: 6333                          # Qdrant port (default: from FLOW_QDRANT_PORT env var or 6333)
      batch_size: 1024                    # Batch size for operations (default: 1024)
      distance: "COSINE"                  # Distance metric: "COSINE", "EUCLIDEAN", or "DOT" (default: "COSINE")
```

##### Qdrant Cloud or Remote Server
```yaml
vector_store:
  default:
    backend: qdrant
    embedding_model: default
    params:
      url: "https://your-cluster.qdrant.io:6333"  # Qdrant server URL (if provided, host and port are ignored)
      api_key: "your-api-key"                     # API key for Qdrant Cloud authentication
      batch_size: 1024                            # Batch size for operations (default: 1024)
      distance: "COSINE"                          # Distance metric (default: "COSINE")
```

#### Configuration Parameters

- **`url`** (optional): Complete URL for connecting to Qdrant. If provided, `host` and `port` are ignored. Useful for Qdrant Cloud or custom deployments
- **`host`** (optional): Host address of the Qdrant server. Defaults to the `FLOW_QDRANT_HOST` environment variable or `"localhost"` if not set
- **`port`** (optional): Port number of the Qdrant server. Defaults to the `FLOW_QDRANT_PORT` environment variable or `6333` if not set
- **`api_key`** (optional): API key for authentication (required for Qdrant Cloud). Can also be set via `FLOW_QDRANT_API_KEY` environment variable
- **`distance`** (optional): Distance metric for vector similarity. Valid values: `"COSINE"`, `"EUCLIDEAN"`, `"DOT"`. Default: `"COSINE"`
- **`batch_size`** (optional): Batch size for bulk operations. Default: `1024`

#### 🌟 Key Features

- **Native Async Support** - All operations have async equivalents for better concurrency
- **Upsert Operations** - Insert automatically updates existing nodes with the same ID
- **Advanced Filtering** - Support for term and range filters on metadata
- **High Performance** - Optimized for large-scale vector similarity search
- **Horizontal Scaling** - Supports clustering for distributed deployments
- **Multiple Distance Metrics** - Cosine, Euclidean, and Dot Product similarity
- **Persistent Storage** - Data is automatically persisted to disk
- **Efficient Iteration** - Scroll through large collections with pagination

### 5. MemoryVectorStore (`backend=memory`)

An ultra-fast in-memory vector store that keeps all data in RAM for maximum performance.

#### 💡 When to Use
- **Testing and development** - Fastest possible operations for unit tests 🧪
- **Small to medium datasets** that fit in memory (< 1M vectors) 💾
- **Applications requiring ultra-low latency** search operations ⚡
- **Temporary workspaces** that don't need persistence 🚀

#### ⚙️ Configuration

```yaml
vector_store:
  default:
    backend: memory
    embedding_model: default
    params:
      store_dir: "./memory_vector_store"  # Directory for backup/restore operations (default: "./memory_vector_store")
      batch_size: 1024                     # Batch size for operations (default: 1024)
```

#### Configuration Parameters

- **`store_dir`** (optional): Directory path for backup/restore operations. Default: `"./memory_vector_store"`
- **`batch_size`** (optional): Batch size for bulk operations. Default: `1024`

#### ⚡ Performance Benefits

- **Zero I/O latency** - All operations happen in RAM
- **Instant search results** - No disk or network overhead
- **Perfect for testing** - Fast setup and teardown
- **Memory efficient** - Only stores what you need

#### 🚨 Important Notes

- **Data is volatile** - Lost when process ends unless explicitly saved
- **Memory usage** - Entire dataset must fit in available RAM
- **No persistence** - Use `dump_workspace()` to save to disk
- **Single process** - Not suitable for distributed applications

## 📝 Example Configurations

### Minimal Configuration (Memory Store)
```yaml
vector_store:
  default:
    backend: memory
    embedding_model: default
```

### Local File Storage
```yaml
vector_store:
  default:
    backend: local
    embedding_model: default
    params:
      store_dir: "./my_vector_store"
      batch_size: 2048
```

### Elasticsearch Production Setup
```yaml
vector_store:
  default:
    backend: elasticsearch
    embedding_model: default
    params:
      hosts: "http://elasticsearch.example.com:9200"
      basic_auth: ["username", "password"]
      batch_size: 2048
```

### Qdrant Cloud Setup
```yaml
vector_store:
  default:
    backend: qdrant
    embedding_model: default
    params:
      url: "https://your-cluster.qdrant.io:6333"
      api_key: "your-api-key-here"
      distance: "COSINE"
      batch_size: 1024
```

## 🔄 Environment Variables

Some vector store backends support environment variables for configuration:

- **Elasticsearch**: `FLOW_ES_HOSTS` - Elasticsearch host(s)
- **Qdrant**:
  - `FLOW_QDRANT_HOST` - Qdrant host (default: "localhost")
  - `FLOW_QDRANT_PORT` - Qdrant port (default: 6333)
  - `FLOW_QDRANT_API_KEY` - Qdrant API key for authentication

Environment variables are used as fallbacks when parameters are not explicitly set in the YAML configuration.

## 🧩 Integration with Embedding Models

All vector stores require an embedding model configuration. The `embedding_model` field in the vector store configuration references a model defined in the `embedding_model` section of `default.yaml`:

```yaml
embedding_model:
  default:
    backend: openai_compatible
    model_name: text-embedding-v4
    params:
      dimensions: 1024

vector_store:
  default:
    backend: memory
    embedding_model: default  # References the embedding_model.default configuration
```

The embedding model configuration provides the model name, backend, and parameters needed for generating vector embeddings.
