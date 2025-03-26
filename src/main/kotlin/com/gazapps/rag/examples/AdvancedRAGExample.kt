package com.gazapps.rag.examples

import com.gazapps.rag.*
import com.gazapps.rag.core.*
import com.gazapps.rag.core.document.ChunkingStrategy
import com.gazapps.rag.core.embedder.OpenAIEmbedder
import com.gazapps.rag.core.embedder.CachedMultiLevelEmbedder
import com.gazapps.rag.core.embedder.MultiLevelEmbeddingCache
import com.gazapps.rag.core.llm.AnthropicClient
import com.gazapps.rag.core.llm.OpenAIClient
import com.gazapps.rag.core.reranking.MaximalMarginalRelevanceReranker
import com.gazapps.rag.core.vectorstore.InMemoryVectorStore
import kotlinx.coroutines.runBlocking
import java.io.File

/**
 * Advanced example showing more complex RAG configurations and features
 * 
 * This example demonstrates:
 * 1. Creating a robust RAG instance with fallbacks
 * 2. Using a cached embedder for performance
 * 3. Configuring advanced chunking strategies
 * 4. Processing a directory of files
 * 5. Using custom reranking
 */
object AdvancedRAGExample {
    @JvmStatic
    fun main(args: Array<String>) {
        // Replace with your actual API keys
        val openAIKey = System.getenv("OPENAI_API_KEY") ?: "your-openai-api-key" 
        val anthropicKey = System.getenv("ANTHROPIC_API_KEY") ?: "your-anthropic-api-key"
        
        // Create a cached embedder for improved performance
        val embeddingCache = MultiLevelEmbeddingCache()
        val cachedEmbedder = CachedMultiLevelEmbedder(
            delegate = OpenAIEmbedder(openAIKey)
            // CachedMultiLevelEmbedder já usa implicitamente o cache
        )
        
        // Create a custom reranker
        val reranker = MaximalMarginalRelevanceReranker()
        
        // Use InMemoryVectorStore for the example since ChromaDB might not be running
        val vectorStore = InMemoryVectorStore()
        
        // Alternative ChromaDB setup (commented out as it requires a running ChromaDB instance)
        // val vectorStore = ChromaDBStore(
        //     baseUrl = "http://localhost:8000",
        //     collectionName = "kotlin-rag-example"
        // )
        
        // Create a robust RAG instance with fallbacks
        val rag = kotlinRag {
            // Configure primary components
            embedder(cachedEmbedder)
            vectorStore(vectorStore)
            llmClient(AnthropicClient(anthropicKey))
            
            // Configure fallbacks
            fallbackEmbedder(OpenAIEmbedder(openAIKey))
            fallbackLLMClient(OpenAIClient(openAIKey))
            
            // Enable error handling explicitly
            withErrorHandling()
            
            // Configure advanced settings
            config {
                // Advanced chunking configuration
                indexing.chunkSize = 300
                indexing.chunkOverlap = 30
                indexing.chunkingStrategy = ChunkingStrategy.SEMANTIC
                indexing.preprocessText = true
                
                // Advanced retrieval configuration
                retrieval.retrievalLimit = 5
                retrieval.reranking = true
                
                // Advanced generation configuration
                generation.includeMetadata = true
                generation.metadataTemplate = "Source: {source}, Date: {date}"
            }
        }
        
        runBlocking {
            // Process a directory of documents
            val docsDirectory = File("./docs")
            if (docsDirectory.exists() && docsDirectory.isDirectory) {
                println("Indexing documents from ${docsDirectory.absolutePath}...")
                
                val results = rag.indexDirectory(
                    directory = docsDirectory,
                    recursive = true,
                    fileExtensions = setOf("pdf", "txt", "docx", "md"),
                    metadata = mapOf("batch" to "example-run")
                )
                
                val successful = results.count { it.value }
                println("Indexed $successful/${results.size} documents successfully")
            } else {
                // If no directory exists, index some sample text
                println("No docs directory found. Indexing sample text...")
                
                val topics = listOf(
                    "Kotlin coroutines provide a way to write asynchronous, non-blocking code in a sequential manner.",
                    "Kotlin's extension functions allow you to extend existing classes with new functionality without inheritance.",
                    "Kotlin's null safety features help prevent null pointer exceptions through nullable types and safe calls.",
                    "Kotlin multiplatform allows sharing code between different platforms like JVM, JavaScript, and Native."
                )
                
                topics.forEachIndexed { index, content ->
                    rag.indexText(
                        content = content,
                        id = "topic-$index",
                        metadata = mapOf("source" to "kotlin-topics", "topic" to "kotlin-features")
                    )
                }
                
                println("Indexed ${topics.size} sample topics")
            }
            
            // Ask a complex question requiring retrieval of multiple documents
            val response = rag.ask(
                question = "What are the key features that make Kotlin productive for developers?",
                options = QueryOptions(
                    retrievalLimit = 4,
                    rerank = true,
                    includeMetadata = true
                )
            )
            
            println("\nQuestion: What are the key features that make Kotlin productive for developers?")
            println("Answer: ${response.answer}")
            println("Retrieved ${response.documents.size} documents")
            
            // Print document snippets and scores
            response.documents.forEachIndexed { index, doc ->
                println("\nDocument ${index + 1} (Score: ${doc.score}):")
                println("${doc.document.content.take(100)}...")
            }
        }
    }
}
