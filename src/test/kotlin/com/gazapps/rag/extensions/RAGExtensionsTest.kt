package com.gazapps.rag.extensions

import com.gazapps.rag.core.*
import com.gazapps.rag.core.document.SimpleDocument
import kotlinx.coroutines.runBlocking
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.io.TempDir
import java.io.File
import java.nio.file.Path

class RAGExtensionsTest {

    private lateinit var mockRag: MockRAG
    
    @BeforeEach
    fun setup() {
        mockRag = MockRAG()
    }
    
    @Test
    fun `indexText should wrap indexFromText in Result`() = runBlocking {
        // Setup
        mockRag.nextIndexResult = true
        
        // Test
        val result = mockRag.indexText("Test content", "test-id")
        
        // Verify
        assertTrue(result.isSuccess)
        assertEquals(true, result.getOrNull())
        assertEquals("Test content", mockRag.lastIndexedContent)
        assertEquals("test-id", mockRag.lastIndexedId)
    }
    
    @Test
    fun `indexText should handle errors with Result`() = runBlocking {
        // Setup
        mockRag.shouldThrow = true
        mockRag.exceptionMessage = "Simulated indexing error"
        
        // Test
        val result = mockRag.indexText("Test content")
        
        // Verify
        assertTrue(result.isFailure)
        assertEquals("Simulated indexing error", result.exceptionOrNull()?.message)
    }
    
    @Test
    fun `ask should wrap query in Result`() = runBlocking {
        // Setup
        val expectedResponse = RAGResponse("Test answer", emptyList())
        mockRag.nextQueryResponse = expectedResponse
        
        // Test
        val result = mockRag.ask("Test question")
        
        // Verify
        assertTrue(result.isSuccess)
        assertEquals(expectedResponse, result.getOrNull())
        assertEquals("Test question", mockRag.lastQueryQuestion)
    }
    
    @Test
    fun `ask should handle errors with Result`() = runBlocking {
        // Setup
        mockRag.shouldThrow = true
        mockRag.exceptionMessage = "Simulated query error"
        
        // Test
        val result = mockRag.ask("Test question")
        
        // Verify
        assertTrue(result.isFailure)
        assertEquals("Simulated query error", result.exceptionOrNull()?.message)
    }
    
    @Test
    fun `String toDocument should create a Document with correct fields`() {
        // Test
        val content = "Test document content"
        val doc = content.toDocument(id = "test-id", metadata = mapOf("source" to "test"))
        
        // Verify
        assertEquals("test-id", doc.id)
        assertEquals(content, doc.content)
        assertEquals("test", doc.metadata["source"])
    }
    
    @Test
    fun `String toDocument should generate id when not provided`() {
        // Test
        val content = "Test document content"
        val doc = content.toDocument(metadata = mapOf("source" to "test"))
        
        // Verify
        assertTrue(doc.id.startsWith("doc-"))
        assertEquals(content, doc.content)
    }
    
    @Test
    fun `File toDocument should create a Document with file metadata`(@TempDir tempDir: Path) {
        // Setup
        val file = File(tempDir.toFile(), "test.txt")
        file.writeText("Test file content")
        
        // Test
        val doc = file.toDocument(metadata = mapOf("tag" to "test-tag"))
        
        // Verify
        assertEquals(file.name, doc.id)
        assertEquals("Test file content", doc.content)
        assertEquals("test.txt", doc.metadata["filename"])
        assertTrue(doc.metadata.containsKey("filesize"))
        assertTrue(doc.metadata.containsKey("filepath"))
        assertTrue(doc.metadata.containsKey("lastModified"))
        assertEquals("test-tag", doc.metadata["tag"])
    }
    
    @Test
    fun `indexFile should use document extractors`(@TempDir tempDir: Path) = runBlocking {
        // Setup
        val file = File(tempDir.toFile(), "test.txt")
        file.writeText("Test file content")
        mockRag.nextIndexResult = true
        
        // Test
        val result = mockRag.indexFile(file, mapOf("tag" to "test-tag"))
        
        // Verify
        assertTrue(result.isSuccess)
        assertEquals(true, result.getOrNull())
        // We would verify more details of the document indexing logic if this
        // wasn't using a mock implementation
    }
    
    // Mock implementation of IRAG for testing
    private class MockRAG : IRAG {
        var lastIndexedContent: String? = null
        var lastIndexedId: String? = null
        var lastQueryQuestion: String? = null
        var nextIndexResult: Boolean = true
        var nextQueryResponse: RAGResponse = RAGResponse("", emptyList())
        var shouldThrow: Boolean = false
        var exceptionMessage: String = "Mock exception"
        
        override suspend fun indexDocument(document: Document): Boolean {
            if (shouldThrow) throw RuntimeException(exceptionMessage)
            lastIndexedContent = document.content
            lastIndexedId = document.id
            return nextIndexResult
        }
        
        override suspend fun indexDocuments(documents: List<Document>): List<Boolean> {
            if (shouldThrow) throw RuntimeException(exceptionMessage)
            return documents.map { 
                lastIndexedContent = it.content
                lastIndexedId = it.id
                nextIndexResult
            }
        }
        
        override suspend fun query(question: String): RAGResponse {
            if (shouldThrow) throw RuntimeException(exceptionMessage)
            lastQueryQuestion = question
            return nextQueryResponse
        }
        
        override suspend fun query(question: String, filter: Map<String, Any>?): RAGResponse {
            if (shouldThrow) throw RuntimeException(exceptionMessage)
            lastQueryQuestion = question
            return nextQueryResponse
        }
        
        override suspend fun query(question: String, options: QueryOptions): RAGResponse {
            if (shouldThrow) throw RuntimeException(exceptionMessage)
            lastQueryQuestion = question
            return nextQueryResponse
        }
        
        override suspend fun indexFromText(
            content: String,
            id: String?,
            metadata: Map<String, Any>,
            chunkContent: Boolean
        ): Boolean {
            if (shouldThrow) throw RuntimeException(exceptionMessage)
            lastIndexedContent = content
            lastIndexedId = id
            return nextIndexResult
        }
        
        override suspend fun indexDocumentWithChunking(document: Document): Boolean {
            if (shouldThrow) throw RuntimeException(exceptionMessage)
            lastIndexedContent = document.content
            lastIndexedId = document.id
            return nextIndexResult
        }
        
        override suspend fun indexFromFile(
            filePath: String,
            metadata: Map<String, Any>,
            chunkContent: Boolean
        ): Boolean {
            if (shouldThrow) throw RuntimeException(exceptionMessage)
            return nextIndexResult
        }
    }
}
