package com.gazapps.rag.core

import kotlinx.coroutines.runBlocking
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue
import kotlin.test.assertFalse
import java.io.File
import kotlin.io.path.createTempFile
import com.gazapps.rag.core.document.*

class RAGTest {
    
    @Test
    fun `RAG should index and retrieve documents`() = runBlocking {
        // Setup
        val embedder = DummyEmbedder()
        val vectorStore = InMemoryVectorStore()
        val llmClient = MockLLMClient()
        
        val rag = RAG(embedder, vectorStore, llmClient)
        
        val document = SimpleDocument(
            id = "test-doc",
            content = "Kotlin is a modern programming language that makes developers more productive.",
            metadata = mapOf("topic" to "programming")
        )
        
        // Act
        val indexResult = rag.indexDocument(document)
        
        // Assert
        assertTrue(indexResult)
        
        // Act - query for related content
        val response = rag.query("What is Kotlin?")
        
        // Assert
        assertNotNull(response.answer)
        assertTrue(response.documents.isNotEmpty())
        assertEquals("test-doc", response.documents.first().document.id)
    }
    
    @Test
    fun `RAG should use filter when querying`() = runBlocking {
        // Setup
        val embedder = DummyEmbedder()
        val vectorStore = InMemoryVectorStore()
        val llmClient = MockLLMClient()
        
        val rag = RAG(embedder, vectorStore, llmClient)
        
        // Add documents with different topics
        val doc1 = SimpleDocument("doc1", "Kotlin is a programming language", mapOf("topic" to "programming"))
        val doc2 = SimpleDocument("doc2", "Python is a programming language", mapOf("topic" to "programming"))
        val doc3 = SimpleDocument("doc3", "Machine learning is an AI technique", mapOf("topic" to "ai"))
        
        rag.indexDocuments(listOf(doc1, doc2, doc3))
        
        // Act - query with filter
        val response = rag.query("What is a programming language?", mapOf("topic" to "programming"))
        
        // Assert
        assertTrue(response.documents.isNotEmpty())
        assertTrue(response.documents.all { it.document.metadata["topic"] == "programming" })
    }
    
    @Test
    fun `RAG DSL should build a properly configured instance`() = runBlocking {
        // Setup & Act
        val ragInstance = rag {
            embedder = DummyEmbedder()
            vectorStore = InMemoryVectorStore()
            llmClient = MockLLMClient()
            config {
                chunkSize = 300
                retrievalLimit = 3
            }
        }
        
        // Assert
        assertNotNull(ragInstance.embedder)
        assertNotNull(ragInstance.vectorStore)
        assertNotNull(ragInstance.llmClient)
        assertEquals(300, ragInstance.config.chunkSize)
        assertEquals(3, ragInstance.config.retrievalLimit)
    }
    
    @Test
    fun `RAG Builder pattern should work with method chaining`() = runBlocking {
        // Setup & Act
        val ragInstance = ragBuilder()
            .withEmbedder(DummyEmbedder())
            .withVectorStore(InMemoryVectorStore())
            .withLLMClient(MockLLMClient())
            .build()
            
        // Assert
        assertNotNull(ragInstance)
        assertNotNull(ragInstance.embedder)
        assertNotNull(ragInstance.vectorStore)
        assertNotNull(ragInstance.llmClient)
    }
    
    @Test
    fun `RAG should index text content directly`() = runBlocking {
        // Setup
        val embedder = DummyEmbedder()
        val vectorStore = InMemoryVectorStore()
        val llmClient = MockLLMClient()
        
        val rag = RAG(embedder, vectorStore, llmClient)
        
        // Act
        val result = rag.indexFromText(
            content = "RAG is a technique for enhancing LLM responses with external knowledge.",
            metadata = mapOf("source" to "article", "date" to "2023-07-12")
        )
        
        // Assert
        assertTrue(result)
        
        // Verify we can query the indexed content
        val response = rag.query("What is RAG?")
        assertNotNull(response.answer)
        assertTrue(response.documents.isNotEmpty())
    }
    
    @Test
    fun `RAG should use QueryOptions when querying`() = runBlocking {
        // Setup
        val embedder = DummyEmbedder()
        val vectorStore = InMemoryVectorStore()
        val llmClient = MockLLMClient()
        
        val rag = RAG(embedder, vectorStore, llmClient)
        
        // Add documents with different topics
        val doc1 = SimpleDocument("doc1", "Kotlin is a programming language", mapOf("topic" to "programming"))
        val doc2 = SimpleDocument("doc2", "Python is a programming language", mapOf("topic" to "programming"))
        val doc3 = SimpleDocument("doc3", "Machine learning is an AI technique", mapOf("topic" to "ai"))
        
        rag.indexDocuments(listOf(doc1, doc2, doc3))
        
        // Act - query with options
        val response = rag.query("What is programming?", QueryOptions(
            filter = mapOf("topic" to "programming"),
            retrievalLimit = 1,
            includeMetadata = false
        ))
        
        // Assert
        assertEquals(1, response.documents.size)
        assertTrue(response.documents.all { it.document.metadata["topic"] == "programming" })
        assertTrue(response.processingTimeMs >= 0)
        assertEquals(response.metadata["documentsRetrieved"], 1)
    }
    
    @Test
    fun `RAG should index from file`() = runBlocking {
        // Setup
        val embedder = DummyEmbedder()
        val vectorStore = InMemoryVectorStore()
        val llmClient = MockLLMClient()
        
        val rag = RAG(embedder, vectorStore, llmClient)
        
        // Create a temporary file for testing
        val tempFile = createTempFile("rag-test-", ".txt").toFile()
        tempFile.writeText("This is a test document about Kotlin RAG implementation.")
        tempFile.deleteOnExit() // Clean up after test
        
        // Act
        val result = rag.indexFromFile(
            filePath = tempFile.absolutePath,
            metadata = mapOf("category" to "test")
        )
        
        // Assert
        assertTrue(result)
        
        // Verify we can query the indexed content
        val response = rag.query("What is this document about?")
        assertNotNull(response.answer)
        assertTrue(response.documents.isNotEmpty())
        
        // Verify metadata
        val docMetadata = response.documents.first().document.metadata
        assertEquals("test", docMetadata["category"])
        assertEquals(tempFile.name, docMetadata["filename"])
        assertTrue(docMetadata.containsKey("last_modified"))
        assertTrue(docMetadata.containsKey("size"))
    }
    
    @Test
    fun `RAG should index from markdown file with frontmatter`() = runBlocking {
        // Setup
        val embedder = DummyEmbedder()
        val vectorStore = InMemoryVectorStore()
        val llmClient = MockLLMClient()
        
        val rag = RAG(embedder, vectorStore, llmClient)
        
        // Create a temporary markdown file with frontmatter
        val tempFile = createTempFile("rag-test-", ".md").toFile()
        tempFile.writeText("""
            ---
            title: Test Document
            author: Test Author
            date: 2023-01-01
            ---
            
            # Kotlin RAG Implementation
            
            This is a test markdown document about implementing RAG in Kotlin.
        """.trimIndent())
        tempFile.deleteOnExit()
        
        // Act
        val result = rag.indexFromFile(
            filePath = tempFile.absolutePath,
            chunkContent = false // Don't chunk for this test
        )
        
        // Assert
        assertTrue(result)
        
        // Query to get the document
        val response = rag.query("What is this document about?")
        assertTrue(response.documents.isNotEmpty())
        
        // Verify content was extracted properly
        val content = response.documents.first().document.content
        assertTrue(content.contains("Kotlin RAG Implementation"))
        
        // Verify frontmatter metadata was extracted
        val docMetadata = response.documents.first().document.metadata
        assertEquals("Test Document", docMetadata["title"])
        assertEquals("Test Author", docMetadata["author"])
        assertEquals("2023-01-01", docMetadata["date"])
    }
    
    @Test
    fun `RAG should chunk large document when indexing`() = runBlocking {
        // Setup
        val embedder = DummyEmbedder()
        val vectorStore = InMemoryVectorStore()
        val llmClient = MockLLMClient()
        
        // Configure RAG with small chunk size
        val rag = RAG(
            embedder, 
            vectorStore, 
            llmClient,
            RAGConfig(
                chunkSize = 100,
                chunkOverlap = 10,
                chunkingStrategy = ChunkingStrategy.PARAGRAPH
            )
        )
        
        // Create a large document that will be split into chunks
        val largeText = """
            # Introduction
            
            This is a large document that will be split into multiple chunks due to its size.
            
            ## First Section
            
            This section contains information about the first topic.
            It spans multiple sentences to ensure it has enough content.
            
            ## Second Section
            
            This is the second major section of the document.
            It also contains multiple sentences.
            
            ## Third Section
            
            Final section with some concluding remarks and summary information.
            This should ensure we have enough text to generate multiple chunks.
        """.trimIndent()
        
        // Act
        val result = rag.indexFromText(
            content = largeText,
            id = "large-doc",
            metadata = mapOf("source" to "test"),
            chunkContent = true
        )
        
        // Assert
        assertTrue(result)
        
        // Query to verify chunks were created
        val response = rag.query("Tell me about the document sections")
        
        // We should have multiple chunks from the same parent document
        val parentIds = response.documents.map { it.document.metadata["parent_id"] }
        assertTrue(parentIds.contains("large-doc"))
        
        // Verify chunks have proper metadata
        response.documents.forEach { scored ->
            val metadata = scored.document.metadata
            if (metadata["parent_id"] == "large-doc") {
                assertTrue(metadata.containsKey("chunk_index"))
                assertTrue(metadata.containsKey("chunk_strategy"))
                assertEquals("PARAGRAPH", metadata["chunk_strategy"])
            }
        }
    }
}
