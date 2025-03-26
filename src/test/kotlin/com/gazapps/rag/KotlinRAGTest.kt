package com.gazapps.rag

import com.gazapps.rag.core.*
import com.gazapps.rag.core.document.ChunkingStrategy
import kotlinx.coroutines.runBlocking
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.io.TempDir
import java.io.File
import java.nio.file.Path
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

class KotlinRAGTest {

    @Test
    fun `standard factory method should create standard RAG implementation`() {
        // Setup
        val embedder = TestEmbedder()
        val vectorStore = TestVectorStore()
        val llmClient = TestLLMClient()
        
        // Test
        val rag = KotlinRAG.standard(embedder, vectorStore, llmClient)
        
        // Verify
        val impl = rag.getRAGImplementation()
        assertInstanceOf(RAG::class.java, impl)
    }
    
    @Test
    fun `robust factory method should create RAG with error handling implementation`() {
        // Setup
        val embedder = TestEmbedder()
        val vectorStore = TestVectorStore()
        val llmClient = TestLLMClient()
        
        // Test
        val rag = KotlinRAG.robust(embedder, vectorStore, llmClient)
        
        // Verify
        val impl = rag.getRAGImplementation()
        assertTrue(impl.javaClass.simpleName.contains("RAGWithErrorHandling"))
    }
    
    @Test
    fun `kotlinRag DSL should create appropriate implementation based on settings`() {
        // Test - Standard implementation
        val standardRag = kotlinRag {
            embedder(TestEmbedder())
            vectorStore(TestVectorStore())
            llmClient(TestLLMClient())
        }
        
        // Test - Robust implementation
        val robustRag = kotlinRag {
            embedder(TestEmbedder())
            vectorStore(TestVectorStore())
            llmClient(TestLLMClient())
            withErrorHandling()
        }
        
        // Verify
        assertInstanceOf(RAG::class.java, standardRag.getRAGImplementation())
        assertTrue(robustRag.getRAGImplementation().javaClass.simpleName.contains("RAGWithErrorHandling"))
    }
    
    @Test
    fun `ask method should delegate to query method`() = runBlocking {
        // Setup
        val llmClient = TestLLMClient()
        llmClient.response = "Test answer"
        
        val rag = kotlinRag {
            embedder(TestEmbedder())
            vectorStore(TestVectorStore())
            llmClient(llmClient)
        }
        
        // Test
        val response = rag.ask("Test question")
        
        // Verify
        assertEquals("Test answer", response.answer)
        assertEquals("Test question", llmClient.lastPrompt)
    }
    
    @Test
    fun `indexText should delegate to indexFromText`() = runBlocking {
        // Setup
        val embedder = TestEmbedder()
        val vectorStore = TestVectorStore()
        
        val rag = kotlinRag {
            embedder(embedder)
            vectorStore(vectorStore)
            llmClient(TestLLMClient())
        }
        
        // Test
        val result = rag.indexText("Test content", "test-id")
        
        // Verify
        assertTrue(result)
        assertEquals(1, vectorStore.storedDocuments.size)
        assertEquals("Test content", vectorStore.storedDocuments["test-id"]?.content)
    }
    
    @Test
    fun `indexTextAsync should execute asynchronously`() {
        // Setup
        val embedder = TestEmbedder()
        val vectorStore = TestVectorStore()
        
        val rag = kotlinRag {
            embedder(embedder)
            vectorStore(vectorStore)
            llmClient(TestLLMClient())
        }
        
        // Test
        val latch = CountDownLatch(1)
        var asyncResult = false
        
        rag.indexTextAsync("Async test content", "async-id") { result ->
            asyncResult = result
            latch.countDown()
        }
        
        // Verify
        assertTrue(latch.await(5, TimeUnit.SECONDS), "Async operation timed out")
        assertTrue(asyncResult)
        assertEquals(1, vectorStore.storedDocuments.size)
        assertEquals("Async test content", vectorStore.storedDocuments["async-id"]?.content)
    }

    @Test
    fun `indexFile should process and index file content`(@TempDir tempDir: Path) = runBlocking {
        // Setup
        val file = File(tempDir.toFile(), "test.txt")
        file.writeText("Test file content")
        
        val embedder = TestEmbedder()
        val vectorStore = TestVectorStore()
        
        val rag = kotlinRag {
            embedder(embedder)
            vectorStore(vectorStore)
            llmClient(TestLLMClient())
            config {
                chunkingStrategy = ChunkingStrategy.PARAGRAPH
            }
        }
        
        // Test
        val result = rag.indexFile(file)
        
        // Verify
        assertTrue(result)
        assertTrue(vectorStore.storedDocuments.isNotEmpty())
        val storedContent = vectorStore.storedDocuments.values.first().content
        assertTrue(storedContent.contains("Test file content"))
    }
    
    /**
     * Test implementation of Embedder
     */
    private class TestEmbedder : Embedder {
        override suspend fun embed(text: String): FloatArray {
            return FloatArray(5) { 0.1f * it }
        }
        
        override suspend fun batchEmbed(texts: List<String>): List<FloatArray> {
            return texts.map { embed(it) }
        }
    }
    
    /**
     * Test implementation of VectorStore
     */
    private class TestVectorStore : VectorStore {
        val storedDocuments = mutableMapOf<String, Document>()
        val storedEmbeddings = mutableMapOf<String, FloatArray>()
        
        override suspend fun store(document: Document, embedding: FloatArray) {
            storedDocuments[document.id] = document
            storedEmbeddings[document.id] = embedding
        }
        
        override suspend fun batchStore(documents: List<Document>, embeddings: List<FloatArray>) {
            documents.zip(embeddings).forEach { (doc, emb) ->
                store(doc, emb)
            }
        }
        
        override suspend fun search(query: FloatArray, limit: Int, filter: Map<String, Any>?): List<ScoredDocument> {
            return storedDocuments.values
                .map { ScoredDocument(it, 0.9f) }
                .take(limit)
        }
        
        override suspend fun delete(documentId: String) {
            storedDocuments.remove(documentId)
            storedEmbeddings.remove(documentId)
        }
        
        override suspend fun clear() {
            storedDocuments.clear()
            storedEmbeddings.clear()
        }
    }
    
    /**
     * Test implementation of LLMClient
     */
    private class TestLLMClient : LLMClient {
        var response: String = "Default test response"
        var lastPrompt: String? = null
        
        override suspend fun generate(prompt: String): String {
            lastPrompt = prompt
            return response
        }
        
        override suspend fun generate(prompt: String, options: GenerationOptions): String {
            lastPrompt = prompt
            return response
        }
    }
}
