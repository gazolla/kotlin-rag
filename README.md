# Kotlin RAG Library

A library for implementing Retrieval-Augmented Generation (RAG) in Kotlin, focusing on robustness, scalability, and ease of use. Built on the principles of DRY (Don't Repeat Yourself) and KISS (Keep It Simple, Stupid), this library enables developers to create applications that can answer questions based on specific documents.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Kotlin](https://img.shields.io/badge/kotlin-1.8.0-blue.svg)](https://kotlinlang.org)

## ✨ Features

- **Flexible Architecture**: Interfaces with multiple implementations
- **Robust Error Handling**: Circuit breakers, fallbacks, retries, and comprehensive error recovery
- **Performance Optimized**: Caching, batch processing, and vectorization optimizations
- **Idiomatic Kotlin API**: DSLs, extension functions, coroutines, and typed results
- **Modular Design**: Hexagonal architecture for easy component replacement
- **Scalable**: From in-memory testing to production-scale deployments
- **Production Ready**: Monitoring, metrics, and logging built-in

## 📦 Installation

### Gradle (Kotlin DSL)

```kotlin
// build.gradle.kts
repositories {
    mavenCentral()
}

dependencies {
    implementation("com.gazapps:kotlin-rag:1.0.0")
}
```

### Maven

```xml
<!-- pom.xml -->
<dependency>
    <groupId>com.gazapps</groupId>
    <artifactId>kotlin-rag</artifactId>
    <version>1.0.0</version>
</dependency>
```

## 🚀 Quick Start

```kotlin
import com.gazapps.rag.*
import com.gazapps.rag.core.*
import com.gazapps.rag.core.embedder.OpenAIEmbedder
import com.gazapps.rag.core.llm.AnthropicClient
import com.gazapps.rag.core.vectorstore.InMemoryVectorStore

// Create a RAG instance
val rag = kotlinRag {
    // Configure core components
    embedder(OpenAIEmbedder("your-openai-api-key"))
    vectorStore(InMemoryVectorStore())
    llmClient(AnthropicClient("your-anthropic-api-key"))
    
    // Configure behavior
    config {
        indexing.chunkSize = 500
        retrieval.retrievalLimit = 3
        generation.promptTemplate = """
            Context:
            {context}
            
            Question:
            {question}
            
            Answer:
        """.trimIndent()
    }
}

// Index some content
rag.indexText(
    content = "Kotlin is a modern programming language that makes developers more productive.",
    metadata = mapOf("source" to "kotlin-intro")
)

// Ask a question
val response = rag.ask("What is Kotlin?")
println("Answer: ${response.answer}")
```

## 🧩 Core Components

### Documents

Documents are the fundamental unit in the RAG system:

```kotlin
interface Document {
    val id: String
    val content: String
    val metadata: Map<String, Any>
    val chunks: List<Document>?
}
```

### Embedders

Embedders convert text to vector embeddings:

```kotlin
// Available implementations:
val openAIEmbedder = OpenAIEmbedder("your-api-key")
val huggingFaceEmbedder = HuggingFaceEmbedder("your-api-key")
val cachedEmbedder = CachedEmbedder(openAIEmbedder, InMemoryEmbeddingCache())
```

### Vector Stores

Vector stores index and retrieve documents by similarity:

```kotlin
// Available implementations:
val inMemoryStore = InMemoryVectorStore()
val chromaDBStore = ChromaDBStore("http://localhost:8000")
val redisVectorStore = RedisVectorStore(redisClient)
val hnswStore = HnswVectorStore(dimensions = 1536)
```

### LLM Clients

LLM clients generate responses from prompts:

```kotlin
// Available implementations:
val openAIClient = OpenAIClient("your-api-key")
val anthropicClient = AnthropicClient("your-api-key")
```

## 📚 Advanced Usage

### Robust Error Handling

```kotlin
val rag = kotlinRag {
    // Primary components
    embedder(OpenAIEmbedder(openAIKey))
    vectorStore(ChromaDBStore(dbUrl))
    llmClient(AnthropicClient(anthropicKey))
    
    // Enable error handling with fallbacks
    withErrorHandling()
    fallbackEmbedder(HuggingFaceEmbedder(hfKey))
    fallbackLLMClient(OpenAIClient(openAIKey))
}
```

### Asynchronous Processing

```kotlin
// Index a directory of files in the background
rag.indexDirectoryAsync(
    directory = File("/path/to/docs"),
    recursive = true,
    fileExtensions = setOf("pdf", "txt", "docx"),
    onProgress = { path, success -> println("$path: ${if (success) "✓" else "✗"}") },
    onComplete = { results -> println("Completed: ${results.count { it.value }}/${results.size}") }
)
```

### Metadata Filtering

```kotlin
// Add metadata to documents
rag.indexText(
    content = "Content about programming",
    metadata = mapOf("category" to "tech", "author" to "Maria")
)

// Query with metadata filter
val response = rag.ask(
    question = "What is programming?", 
    filter = mapOf("category" to "tech")
)
```

### Custom Chunking

```kotlin
val rag = kotlinRag {
    // Other configuration...
    
    config {
        indexing.chunkSize = 300
        indexing.chunkOverlap = 50
        indexing.chunkingStrategy = ChunkingStrategy.SEMANTIC
    }
}
```

### Advanced Retrieval and Reranking

```kotlin
val response = rag.ask(
    question = "How does Kotlin handle null safety?",
    options = QueryOptions(
        retrievalLimit = 8,
        rerank = true,
        includeMetadata = true
    )
)
```

## 🏗 Architecture

The library follows a hexagonal (ports & adapters) architecture with these core interfaces:

- `Document`: Represents text with metadata
- `Embedder`: Converts text to vector embeddings
- `VectorStore`: Stores and retrieves documents by similarity search
- `LLMClient`: Generates responses using language models
- `IRAG`: Unifies all RAG implementations with a common interface

![Architecture Diagram](docs/images/architecture.png)

## 📊 Monitoring and Metrics

The library includes built-in monitoring capabilities:

```kotlin
// Access metrics
val metrics = RAGMetricsManager.getMetrics()

// Log metrics
println("Avg. embedding time: ${metrics.getAverageTime("embedding.time")}ms")
println("Total queries: ${metrics.getCounter("queries.total")}")
println("Cache hit ratio: ${metrics.getCacheHitRatio()}%")
```

## 🧪 Testing

For testing, use the mock implementations:

```kotlin
val testRag = kotlinRag {
    embedder(MockEmbedder())
    vectorStore(InMemoryVectorStore())
    llmClient(MockLLMClient(predefinedResponses = mapOf(
        "What is Kotlin?" to "Kotlin is a modern programming language."
    )))
}
```

## 📖 Documentation

- [API Reference](docs/api-reference.md)
- [Usage Guides](docs/guides/)
  - [Basic Usage](docs/guides/basic-usage.md)
  - [Advanced Configuration](docs/guides/advanced-configuration.md)
  - [Working with Documents](docs/guides/working-with-documents.md)
  - [Error Handling](docs/guides/error-handling.md)
  - [Performance Optimization](docs/guides/performance-optimization.md)
- [Examples](src/main/kotlin/com/gazapps/rag/examples/)
  - [Basic RAG Example](src/main/kotlin/com/gazapps/rag/examples/BasicRAGExample.kt)
  - [Advanced RAG Example](src/main/kotlin/com/gazapps/rag/examples/AdvancedRAGExample.kt)
  - [Custom Document Example](src/main/kotlin/com/gazapps/rag/examples/CustomDocumentExample.kt)
  - [Error Handling Example](src/main/kotlin/com/gazapps/rag/examples/ErrorHandlingExample.kt)

## 🔧 Configuration Options

| Category | Option | Description | Default |
|----------|--------|-------------|---------|
| Indexing | `chunkSize` | Maximum size of document chunks | 500 |
| Indexing | `chunkOverlap` | Overlap between chunks | 50 |
| Indexing | `chunkingStrategy` | How to chunk documents | `PARAGRAPH` |
| Retrieval | `retrievalLimit` | Number of docs to retrieve | 5 |
| Retrieval | `reranking` | Whether to rerank results | `false` |
| Generation | `promptTemplate` | Template for LLM prompts | *Basic template* |

## 🤝 Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
