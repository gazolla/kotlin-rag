package com.gazapps.rag.core

import kotlinx.coroutines.runBlocking
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class VectorStoreTest {
    
    private val testDocuments = listOf(
        SimpleDocument("doc1", "The quick brown fox jumps over the lazy dog", 
                      mapOf("category" to "animals")),
        SimpleDocument("doc2", "Machine learning models process and learn from data", 
                      mapOf("category" to "technology")),
        SimpleDocument("doc3", "Python and Kotlin are popular programming languages", 
                      mapOf("category" to "technology"))
    )
    
    private val testEmbeddings = listOf(
        floatArrayOf(0.1f, 0.2f, 0.3f),
        floatArrayOf(0.4f, 0.5f, 0.6f),
        floatArrayOf(0.7f, 0.8f, 0.9f)
    ).map { VectorUtils.normalize(it) }
    
    @Test
    fun `InMemoryVectorStore should store and retrieve documents`() = runBlocking {
        // Setup
        val vectorStore = InMemoryVectorStore()
        
        // Act
        vectorStore.store(testDocuments[0], testEmbeddings[0])
        
        // Create a query that should match the first document
        val queryVector = testEmbeddings[0].copyOf()
        val results = vectorStore.search(queryVector, limit = 1)
        
        // Assert
        assertEquals(1, results.size)
        assertEquals("doc1", results[0].document.id)
        assertEquals(1.0f, results[0].score, 0.0001f) // Perfect match should have score of 1.0
    }
    
    @Test
    fun `InMemoryVectorStore should handle batch storage`() = runBlocking {
        // Setup
        val vectorStore = InMemoryVectorStore()
        
        // Act
        vectorStore.batchStore(testDocuments, testEmbeddings)
        
        // Query for technology-related documents
        val queryVector = floatArrayOf(0.6f, 0.7f, 0.8f)
        VectorUtils.normalize(queryVector)
        val results = vectorStore.search(queryVector, limit = 3)
        
        // Assert
        assertEquals(3, results.size)
        // Results should be sorted by similarity score
        assertEquals("doc3", results[0].document.id)
    }
    
    @Test
    fun `InMemoryVectorStore should filter by metadata`() = runBlocking {
        // Setup
        val vectorStore = InMemoryVectorStore()
        vectorStore.batchStore(testDocuments, testEmbeddings)
        
        // Act - search with a filter for technology category
        val queryVector = floatArrayOf(0.5f, 0.5f, 0.5f)
        VectorUtils.normalize(queryVector)
        val results = vectorStore.search(
            query = queryVector,
            limit = 10,
            filter = mapOf("category" to "technology")
        )
        
        // Assert
        assertEquals(2, results.size) // Should only return the 2 technology documents
        assertTrue(results.all { it.document.metadata["category"] == "technology" })
    }
    
    @Test
    fun `InMemoryVectorStore should delete documents`() = runBlocking {
        // Setup
        val vectorStore = InMemoryVectorStore()
        vectorStore.batchStore(testDocuments, testEmbeddings)
        
        // Act
        vectorStore.delete("doc1")
        val queryVector = testEmbeddings[0].copyOf() // Should match doc1 if it existed
        val results = vectorStore.search(queryVector, limit = 3)
        
        // Assert
        assertTrue(results.none { it.document.id == "doc1" })
        assertEquals(2, results.size)
    }
    
    @Test
    fun `InMemoryVectorStore should clear all documents`() = runBlocking {
        // Setup
        val vectorStore = InMemoryVectorStore()
        vectorStore.batchStore(testDocuments, testEmbeddings)
        
        // Act
        vectorStore.clear()
        val queryVector = floatArrayOf(0.5f, 0.5f, 0.5f)
        VectorUtils.normalize(queryVector)
        val results = vectorStore.search(queryVector, limit = 10)
        
        // Assert
        assertEquals(0, results.size)
    }
    
    @Test
    fun `InMemoryVectorStore size method should return correct count`() = runBlocking {
        // Setup
        val vectorStore = InMemoryVectorStore()
        
        // Act & Assert
        assertEquals(0, vectorStore.size())
        
        vectorStore.store(testDocuments[0], testEmbeddings[0])
        assertEquals(1, vectorStore.size())
        
        vectorStore.batchStore(testDocuments.subList(1, 3), testEmbeddings.subList(1, 3))
        assertEquals(3, vectorStore.size())
        
        vectorStore.delete("doc1")
        assertEquals(2, vectorStore.size())
        
        vectorStore.clear()
        assertEquals(0, vectorStore.size())
    }
    
    @Test
    fun `InMemoryVectorStore should throw exception when dimensions don't match`() = runBlocking {
        // Setup
        val vectorStore = InMemoryVectorStore()
        vectorStore.store(testDocuments[0], floatArrayOf(0.1f, 0.2f, 0.3f))
        
        // Act & Assert
        try {
            vectorStore.search(floatArrayOf(0.1f, 0.2f), limit = 1)
            throw AssertionError("Should have thrown an exception for mismatched dimensions")
        } catch (e: IllegalArgumentException) {
            // Expected
            assertTrue(e.message?.contains("dimensions") == true)
        }
    }
    
    @Test
    fun `VectorStore factory should create different similarity metrics`() = runBlocking {
        // Setup
        val cosineStore = InMemoryVectorStore.withSimilarityMetric(
            InMemoryVectorStore.Companion.SimilarityMetric.COSINE
        )
        val dotProductStore = InMemoryVectorStore.withSimilarityMetric(
            InMemoryVectorStore.Companion.SimilarityMetric.DOT_PRODUCT
        )
        val euclideanStore = InMemoryVectorStore.withSimilarityMetric(
            InMemoryVectorStore.Companion.SimilarityMetric.EUCLIDEAN
        )
        
        // They should all work with the same API
        cosineStore.store(testDocuments[0], testEmbeddings[0])
        dotProductStore.store(testDocuments[0], testEmbeddings[0])
        euclideanStore.store(testDocuments[0], testEmbeddings[0])
        
        // Just check that they can be searched
        assertEquals(1, cosineStore.search(testEmbeddings[0], 1).size)
        assertEquals(1, dotProductStore.search(testEmbeddings[0], 1).size)
        assertEquals(1, euclideanStore.search(testEmbeddings[0], 1).size)
    }
}
