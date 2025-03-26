# Kotlin RAG Library

Uma biblioteca moderna e poderosa para implementação de Retrieval-Augmented Generation (RAG) em Kotlin, com foco em robustez, escalabilidade e facilidade de uso.

## Características

- **Flexibilidade**: Interfaces bem definidas com múltiplas implementações
- **Robustez**: Tratamento de erros abrangente, fallbacks, e monitoramento
- **Desempenho**: Caching, processamento em lotes, e otimizações
- **API Kotlin Idiomática**: DSLs fluentes, funções de extensão, e tipo Result
- **Desacoplamento**: Arquitetura hexagonal para fácil substituição de componentes

## Instalação

```kotlin
// build.gradle.kts
dependencies {
    implementation("com.gazapps:kotlin-rag:1.0.0")
}
```

## Uso Básico

```kotlin
import com.gazapps.rag.*
import com.gazapps.rag.core.*
import com.gazapps.rag.extensions.*

// Criar uma instância RAG
val rag = kotlinRag {
    embedder(OpenAIEmbedder("seu-api-key"))
    vectorStore(InMemoryVectorStore())
    llmClient(AnthropicClient("seu-api-key"))
    
    config {
        chunkSize = 500
        retrievalLimit = 3
        promptTemplate = "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    }
}

// Indexar documentos
rag.indexText("Kotlin é uma linguagem moderna e expressiva.")
rag.indexFile(File("documento.pdf"))

// Fazer perguntas
val resposta = rag.ask("O que é Kotlin?")
println(resposta.fold(
    onSuccess = { "Resposta: ${it.answer}" },
    onFailure = { "Erro: ${it.message}" }
))
```

## Características Avançadas

### Tratamento de Erros Robusto

```kotlin
val rag = kotlinRag {
    // Componentes primários
    embedder(OpenAIEmbedder(apiKey))
    vectorStore(ChromaDBStore(dbUrl))
    llmClient(AnthropicClient(apiKey))
    
    // Habilitar tratamento de erros com fallbacks
    withErrorHandling()
    fallbackEmbedder(HuggingFaceEmbedder(hfKey))
    fallbackLLMClient(OpenAIClient(openaiKey))
}
```

### Processamento Assíncrono

```kotlin
// Indexar um diretório em background
rag.indexDirectoryAsync(
    directory = File("/caminho/docs"),
    recursive = true,
    fileExtensions = setOf("pdf", "txt", "doc"),
    onProgress = { path, success -> println("$path: ${if (success) "✓" else "✗"}") },
    onComplete = { results -> println("Concluído: ${results.count { it.value }}/${results.size}") }
)
```

### Filtragem por Metadados

```kotlin
// Adicionar metadados aos documentos
rag.indexText(
    content = "Conteúdo sobre programação",
    metadata = mapOf("categoria" to "tech", "autor" to "Maria")
)

// Consultar com filtro de metadados
val resposta = rag.ask("O que é programação?", mapOf("categoria" to "tech"))
```

## Arquitetura

A biblioteca segue uma arquitetura hexagonal (ports & adapters) com as seguintes interfaces principais:

- `Embedder`: Converte texto em vetores de embedding
- `VectorStore`: Armazena e recupera documentos por similaridade vetorial
- `LLMClient`: Gera respostas usando modelos de linguagem
- `IRAG`: Interface unificada para todas as implementações RAG

## Implementações Disponíveis

### Embedders
- `OpenAIEmbedder`: Usa a API de embeddings da OpenAI
- `HuggingFaceEmbedder`: Usa modelos de embedding do HuggingFace
- `LocalEmbedder`: Embeddings locais para ambientes offline

### Vector Stores
- `InMemoryVectorStore`: Armazenamento em memória para testes
- `ChromaDBStore`: Integração com ChromaDB
- `RedisVectorStore`: Usar Redis como vector store
- `HnswVectorStore`: Implementação local eficiente baseada em HNSW

### LLM Clients
- `OpenAIClient`: Integração com modelos GPT da OpenAI
- `AnthropicClient`: Integração com modelos Claude da Anthropic
- `HuggingFaceClient`: Integração com modelos do HuggingFace

## Licença

MIT
