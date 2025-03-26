package com.gazapps.rag

import com.gazapps.rag.core.*
import com.gazapps.rag.core.document.ChunkingStrategy
import com.gazapps.rag.core.error.*
import com.gazapps.rag.core.monitoring.RAGMetricsManager
import kotlinx.coroutines.runBlocking

/**
 * Simple demonstration of RAG with error handling capabilities
 */
fun main() = runBlocking {
    println("=== Kotlin RAG Library - Error Handling Demo ===")
    
    // Setup logging
    val logger = RAGLoggerFactory.createConsoleLogger(LogLevel.INFO)
    RAGLoggerFactory.setLogger(logger)
    
    // Create a simple in-memory example
    val rag = ragWithErrorHandling {
        // Use mock components for the example
        embedder = object : Embedder {
            var failCount = 0
            
            override suspend fun embed(text: String): FloatArray {
                if (failCount++ % 3 == 0) {
                    logger.info("Simulating embedder failure", "Main")
                    throw RAGException.EmbeddingException("Simulated embedder failure")
                }
                return FloatArray(10) { 0.1f * it }
            }
            
            override suspend fun batchEmbed(texts: List<String>): List<FloatArray> {
                return texts.map { embed(it) }
            }
        }
        
        fallbackEmbedder = object : Embedder {
            override suspend fun embed(text: String): FloatArray {
                logger.info("Using fallback embedder", "Main")
                return FloatArray(10) { 0.2f * it }
            }
            
            override suspend fun batchEmbed(texts: List<String>): List<FloatArray> {
                return texts.map { embed(it) }
            }
        }
        
        vectorStore = object : VectorStore {
            private val docs = mutableMapOf<String, Pair<Document, FloatArray>>()
            
            override suspend fun store(document: Document, embedding: FloatArray) {
                docs[document.id] = Pair(document, embedding)
            }
            
            override suspend fun batchStore(documents: List<Document>, embeddings: List<FloatArray>) {
                documents.zip(embeddings).forEach { (doc, emb) -> store(doc, emb) }
            }
            
            override suspend fun search(query: FloatArray, limit: Int, filter: Map<String, Any>?): List<ScoredDocument> {
                return docs.values.map { (doc, _) -> 
                    ScoredDocument(doc, 0.8f)
                }.take(limit)
            }
            
            override suspend fun delete(documentId: String) {
                docs.remove(documentId)
            }
            
            override suspend fun clear() {
                docs.clear()
            }
        }
        
        llmClient = object : LLMClient {
            override suspend fun generate(prompt: String): String {
                return "This is a response to: ${prompt.takeLast(50)}..."
            }
            
            override suspend fun generate(prompt: String, options: GenerationOptions): String {
                return generate(prompt)
            }
        }
        
        config {
            chunkSize = 100
            chunkingStrategy = ChunkingStrategy.PARAGRAPH
            retrievalLimit = 2
        }
    }
    
    // Index some sample content
    println("\nIndexing sample documents:")
    
    val documents = listOf(
        "RAG (Retrieval-Augmented Generation) is a technique that combines retrieval systems with generative AI.",
        "Error handling in Kotlin can be done using try-catch blocks, exceptions, and the Result type.",
        "Circuit breakers prevent cascading failures in distributed systems.",
        "Metrics and logging are essential for monitoring application health."
    )
    
    documents.forEachIndexed { index, content ->
        val result = rag.indexFromText(content, "doc-$index")
        println("Document $index: ${if (result) "Success ✓" else "Failed ✗"}")
    }
    
    // Run some sample queries
    println("\nExecuting queries with error handling:")
    val queries = listOf(
        "What is RAG?",
        "How do circuit breakers work?",
        "Why are metrics important?",
        "What is error handling in Kotlin?"
    )
    
    queries.forEach { query ->
        println("\nQuery: $query")
        try {
            val response = rag.query(query)
            println("Answer: ${response.answer}")
            println("Documents: ${response.documents.size}")
            println("Processing time: ${response.metadata["processingTimeMs"]} ms")
        } catch (e: Exception) {
            println("Error: ${e.message}")
        }
    }
    
    // Print metrics
    println("\nMetrics Report:")
    println(RAGMetricsManager.getMetrics().getMetricsReport())
}
