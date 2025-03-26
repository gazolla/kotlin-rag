package com.gazapps.rag.core

import com.gazapps.rag.core.document.ChunkingStrategy
import com.gazapps.rag.core.document.SimpleDocument
import com.gazapps.rag.core.error.*
import com.gazapps.rag.core.monitoring.RAGMetrics
import kotlinx.coroutines.runBlocking
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.io.TempDir
import java.io.File
import java.io.IOException
import java.nio.file.Path

class RAGWithErrorHandlingTest {
    
    private lateinit var mockEmbedder: MockFailingEmbedder
    private lateinit var mockVectorStore: MockFailingVectorStore
    private lateinit var mockLLMClient: MockFailingLLMClient
    
    private lateinit var fallbackEmbedder: MockEmbedder
    private lateinit var fallbackVectorStore: MockVectorStore
    private lateinit var fallbackLLMClient: MockLLMClient
    
    private lateinit var metrics: RAGMetrics
    private lateinit var logger: TestLogger
    
    private lateinit var rag: RAGWithErrorHandling
    
    @BeforeEach
    fun setup() {
        mockEmbedder = MockFailingEmbedder()
        mockVectorStore = MockFailingVectorStore()
        mockLLMClient = MockFailingLLMClient()
        
        fallbackEmbedder = MockEmbedder()
        fallbackVectorStore = MockVectorStore()
        fallbackLLMClient = MockLLMClient()
        
        metrics = RAGMetrics()
        logger = TestLogger()
        
        rag = ragWithErrorHandling {
            embedder = mockEmbedder
            vectorStore = mockVectorStore
            llmClient = mockLLMClient
            fallbackEmbedder = this@RAGWithErrorHandlingTest.fallbackEmbedder
            fallbackVectorStore = this@RAGWithErrorHandlingTest.fallbackVectorStore
            fallbackLLMClient = this@RAGWithErrorHandlingTest.fallbackLLMClient
            metrics = this@RAGWithErrorHandlingTest.metrics
            logger = this@RAGWithErrorHandlingTest.logger
            config {
                chunkSize = 100
                chunkOverlap = 20
                retrievalLimit = 3
                promptTemplate = "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
                chunkingStrategy = ChunkingStrategy.PARAGRAPH
            }
        }
    }
    
    @Test
    fun `index document should succeed with primary components`() = runBlocking {
        // Configure mock components to work
        mockEmbedder.shouldFail = false
        mockVectorStore.shouldFail = false
        
        // Create test document
        val doc = SimpleDocument("test-1", "Test content", mapOf("source" to "test"))
        
        // Index document
        val result = rag.indexDocument(doc)
        
        // Verify results
        assertTrue(result)
        assertEquals(1, mockEmbedder.embedCalls.size)
        assertEquals("Test content", mockEmbedder.embedCalls[0])
        
        assertEquals(1, mockVectorStore.storeCalls.size)
        assertEquals(doc, mockVectorStore.storeCalls[0].first)
        
        // Verify metrics
        assertEquals(1, metrics.getCounter("rag.index.document.attempts"))
        assertEquals(1, metrics.getCounter("rag.index.document.success"))
        
        // Verify logs
        assertTrue(logger.logs.any { it.level == LogLevel.INFO && it.message.contains("Document indexed successfully") })
    }
    
    @Test
    fun `index document should use fallback embedder when primary fails`() = runBlocking {
        // Configure primary to fail, fallback to work
        mockEmbedder.shouldFail = true
        mockVectorStore.shouldFail = false
        
        // Create test document
        val doc = SimpleDocument("test-1", "Test content", mapOf("source" to "test"))
        
        // Index document
        val result = rag.indexDocument(doc)
        
        // Verify results
        assertTrue(result)
        assertEquals(3, mockEmbedder.embedCalls.size) // Initial + 2 retries
        assertEquals(1, fallbackEmbedder.embedCalls.size)
        assertEquals("Test content", fallbackEmbedder.embedCalls[0])
        
        assertEquals(1, mockVectorStore.storeCalls.size)
        assertEquals(doc, mockVectorStore.storeCalls[0].first)
        
        // Verify metrics
        assertEquals(1, metrics.getCounter("rag.index.document.attempts"))
        assertEquals(1, metrics.getCounter("rag.index.document.success"))
        
        // Verify logs
        assertTrue(logger.logs.any { it.level == LogLevel.WARN && it.message.contains("Using fallback embedder") })
    }
    
    @Test
    fun `index document should use fallback vector store when primary fails`() = runBlocking {
        // Configure primary embedder to work, primary vector store to fail
        mockEmbedder.shouldFail = false
        mockVectorStore.shouldFail = true
        
        // Create test document
        val doc = SimpleDocument("test-1", "Test content", mapOf("source" to "test"))
        
        // Index document
        val result = rag.indexDocument(doc)
        
        // Verify results
        assertTrue(result)
        assertEquals(1, mockEmbedder.embedCalls.size)
        assertEquals("Test content", mockEmbedder.embedCalls[0])
        
        assertEquals(3, mockVectorStore.storeCalls.size) // Initial + 2 retries
        assertEquals(1, fallbackVectorStore.storeCalls.size)
        assertEquals(doc, fallbackVectorStore.storeCalls[0].first)
        
        // Verify logs
        assertTrue(logger.logs.any { it.level == LogLevel.WARN && it.message.contains("Using fallback vector store") })
    }
    
    @Test
    fun `query should succeed with primary components`() = runBlocking {
        // Configure mock components to work
        mockEmbedder.shouldFail = false
        mockVectorStore.shouldFail = false
        mockLLMClient.shouldFail = false
        
        // Mock vector store to return documents
        val docs = listOf(
            SimpleDocument("doc1", "Document 1 content", mapOf("source" to "test")),
            SimpleDocument("doc2", "Document 2 content", mapOf("source" to "test"))
        )
        mockVectorStore.searchResults = docs.map { ScoredDocument(it, 0.9f) }
        
        // Set mock LLM response
        mockLLMClient.response = "This is the answer."
        
        // Query
        val query = "What is the test question?"
        val response = rag.query(query)
        
        // Verify results
        assertEquals("This is the answer.", response.answer)
        assertEquals(2, response.documents.size)
        
        // Verify component calls
        assertEquals(1, mockEmbedder.embedCalls.size)
        assertEquals(query, mockEmbedder.embedCalls[0])
        
        assertEquals(1, mockVectorStore.searchCalls.size)
        
        assertEquals(1, mockLLMClient.generateCalls.size)
        assertTrue(mockLLMClient.generateCalls[0].contains(query))
        
        // Verify metrics
        assertEquals(1, metrics.getCounter("rag.query.attempts"))
        assertEquals(1, metrics.getCounter("rag.query.success"))
        
        // Verify logs
        assertTrue(logger.logs.any { it.level == LogLevel.INFO && it.message.contains("Query processed successfully") })
    }
    
    @Test
    fun `query should use fallbacks when primary components fail`() = runBlocking {
        // Configure all primary components to fail
        mockEmbedder.shouldFail = true
        mockVectorStore.shouldFail = true
        mockLLMClient.shouldFail = true
        
        // Mock fallback vector store to return documents
        val docs = listOf(
            SimpleDocument("doc1", "Document 1 content", mapOf("source" to "test")),
            SimpleDocument("doc2", "Document 2 content", mapOf("source" to "test"))
        )
        fallbackVectorStore.searchResults = docs.map { ScoredDocument(it, 0.9f) }
        
        // Set fallback LLM response
        fallbackLLMClient.response = "This is the fallback answer."
        
        // Query
        val query = "What is the test question?"
        val response = rag.query(query)
        
        // Verify results
        assertEquals("This is the fallback answer.", response.answer)
        assertEquals(2, response.documents.size)
        
        // Verify component calls
        assertTrue(mockEmbedder.embedCalls.size > 1) // Initial + retries
        assertEquals(1, fallbackEmbedder.embedCalls.size)
        assertEquals(query, fallbackEmbedder.embedCalls[0])
        
        assertTrue(mockVectorStore.searchCalls.size > 1) // Initial + retries
        assertEquals(1, fallbackVectorStore.searchCalls.size)
        
        assertTrue(mockLLMClient.generateCalls.size > 1) // Initial + retries
        assertEquals(1, fallbackLLMClient.generateCalls.size)
        
        // Verify logs
        assertTrue(logger.logs.any { it.level == LogLevel.WARN && it.message.contains("Using fallback embedder") })
        assertTrue(logger.logs.any { it.level == LogLevel.WARN && it.message.contains("Using fallback vector store") })
        assertTrue(logger.logs.any { it.level == LogLevel.WARN && it.message.contains("Using fallback LLM") })
    }
    
    @Test
    fun `query should return error response when all components fail`() = runBlocking {
        // Create RAG without fallbacks
        val ragNoFallbacks = ragWithErrorHandling {
            embedder = mockEmbedder
            vectorStore = mockVectorStore
            llmClient = mockLLMClient
            metrics = this@RAGWithErrorHandlingTest.metrics
            logger = this@RAGWithErrorHandlingTest.logger
        }
        
        // Configure all components to fail
        mockEmbedder.shouldFail = true
        
        // Query
        val query = "What is the test question?"
        val response = ragNoFallbacks.query(query)
        
        // Verify error response
        assertTrue(response.answer.contains("unavailable"))
        assertTrue(response.documents.isEmpty())
        assertEquals(true, response.metadata["error"])
        assertNotNull(response.metadata["errorType"])
        assertNotNull(response.metadata["errorMessage"])
        
        // Verify metrics
        assertEquals(1, metrics.getCounter("rag.query.attempts"))
        assertEquals(1, metrics.getCounter("rag.query.failures"))
        
        // Verify logs
        assertTrue(logger.logs.any { it.level == LogLevel.ERROR && it.message.contains("Query processing failed") })
    }
    
    @Test
    fun `circuit breaker should prevent further calls after multiple failures`() = runBlocking {
        // Configure mock embedder to fail
        mockEmbedder.shouldFail = true
        
        // Create RAG without fallbacks
        val ragNoFallbacks = ragWithErrorHandling {
            embedder = mockEmbedder
            vectorStore = mockVectorStore
            llmClient = mockLLMClient
            metrics = this@RAGWithErrorHandlingTest.metrics
            logger = this@RAGWithErrorHandlingTest.logger
        }
        
        // Create test document
        val doc = SimpleDocument("test-1", "Test content", mapOf("source" to "test"))
        
        // Trigger failures to open circuit breaker (5 failures)
        repeat(5) {
            ragNoFallbacks.indexDocument(doc)
        }
        
        // Reset mock calls to verify no more calls are made
        mockEmbedder.embedCalls.clear()
        
        // Try indexing again
        val result = ragNoFallbacks.indexDocument(doc)
        
        // Verify circuit breaker prevented the call
        assertFalse(result)
        assertTrue(mockEmbedder.embedCalls.isEmpty()) // No calls made
        
        // Verify logs
        assertTrue(logger.logs.any { it.level == LogLevel.WARN && it.message.contains("Circuit breaker is open") })
    }
    
    // Mock implementations for testing
    
    private class TestLogger : RAGLogger {
        val logs = mutableListOf<LogEntry>()
        
        override fun log(entry: LogEntry) {
            logs.add(entry)
        }
    }
    
    private class MockEmbedder : Embedder {
        val embedCalls = mutableListOf<String>()
        val batchEmbedCalls = mutableListOf<List<String>>()
        
        override suspend fun embed(text: String): FloatArray {
            embedCalls.add(text)
            return FloatArray(10) { 0.1f * it }
        }
        
        override suspend fun batchEmbed(texts: List<String>): List<FloatArray> {
            batchEmbedCalls.add(texts)
            return texts.map { FloatArray(10) { 0.1f * it } }
        }
    }
    
    private class MockVectorStore : VectorStore {
        val storeCalls = mutableListOf<Pair<Document, FloatArray>>()
        val batchStoreCalls = mutableListOf<Pair<List<Document>, List<FloatArray>>>()
        val searchCalls = mutableListOf<Triple<FloatArray, Int, Map<String, Any>?>>()
        val deleteCalls = mutableListOf<String>()
        
        var searchResults = emptyList<ScoredDocument>()
        
        override suspend fun store(document: Document, embedding: FloatArray) {
            storeCalls.add(document to embedding)
        }
        
        override suspend fun batchStore(documents: List<Document>, embeddings: List<FloatArray>) {
            batchStoreCalls.add(documents to embeddings)
        }
        
        override suspend fun search(query: FloatArray, limit: Int, filter: Map<String, Any>?): List<ScoredDocument> {
            searchCalls.add(Triple(query, limit, filter))
            return searchResults
        }
        
        override suspend fun delete(documentId: String) {
            deleteCalls.add(documentId)
        }
    }
    
    private class MockLLMClient : LLMClient {
        val generateCalls = mutableListOf<String>()
        val generateWithOptionsCalls = mutableListOf<Pair<String, GenerationOptions>>()
        
        var response = "Mock LLM response"
        
        override suspend fun generate(prompt: String): String {
            generateCalls.add(prompt)
            return response
        }
        
        override suspend fun generate(prompt: String, options: GenerationOptions): String {
            generateWithOptionsCalls.add(prompt to options)
            return response
        }
    }
    
    private class MockFailingEmbedder : Embedder {
        val embedCalls = mutableListOf<String>()
        val batchEmbedCalls = mutableListOf<List<String>>()
        
        var shouldFail = false
        var failureMessage = "Simulated embedder failure"
        
        override suspend fun embed(text: String): FloatArray {
            embedCalls.add(text)
            if (shouldFail) {
                throw IOException(failureMessage)
            }
            return FloatArray(10) { 0.1f * it }
        }
        
        override suspend fun batchEmbed(texts: List<String>): List<FloatArray> {
            batchEmbedCalls.add(texts)
            if (shouldFail) {
                throw IOException(failureMessage)
            }
            return texts.map { FloatArray(10) { 0.1f * it } }
        }
    }
    
    private class MockFailingVectorStore : VectorStore {
        val storeCalls = mutableListOf<Pair<Document, FloatArray>>()
        val batchStoreCalls = mutableListOf<Pair<List<Document>, List<FloatArray>>>()
        val searchCalls = mutableListOf<Triple<FloatArray, Int, Map<String, Any>?>>()
        val deleteCalls = mutableListOf<String>()
        
        var searchResults = emptyList<ScoredDocument>()
        var shouldFail = false
        var failureMessage = "Simulated vector store failure"
        
        override suspend fun store(document: Document, embedding: FloatArray) {
            storeCalls.add(document to embedding)
            if (shouldFail) {
                throw IOException(failureMessage)
            }
        }
        
        override suspend fun batchStore(documents: List<Document>, embeddings: List<FloatArray>) {
            batchStoreCalls.add(documents to embeddings)
            if (shouldFail) {
                throw IOException(failureMessage)
            }
        }
        
        override suspend fun search(query: FloatArray, limit: Int, filter: Map<String, Any>?): List<ScoredDocument> {
            searchCalls.add(Triple(query, limit, filter))
            if (shouldFail) {
                throw IOException(failureMessage)
            }
            return searchResults
        }
        
        override suspend fun delete(documentId: String) {
            deleteCalls.add(documentId)
            if (shouldFail) {
                throw IOException(failureMessage)
            }
        }
    }
    
    private class MockFailingLLMClient : LLMClient {
        val generateCalls = mutableListOf<String>()
        val generateWithOptionsCalls = mutableListOf<Pair<String, GenerationOptions>>()
        
        var response = "Mock LLM response"
        var shouldFail = false
        var failureMessage = "Simulated LLM failure"
        
        override suspend fun generate(prompt: String): String {
            generateCalls.add(prompt)
            if (shouldFail) {
                throw IOException(failureMessage)
            }
            return response
        }
        
        override suspend fun generate(prompt: String, options: GenerationOptions): String {
            generateWithOptionsCalls.add(prompt to options)
            if (shouldFail) {
                throw IOException(failureMessage)
            }
            return response
        }
    }
}
