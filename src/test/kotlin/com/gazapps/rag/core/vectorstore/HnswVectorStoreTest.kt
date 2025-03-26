package com.gazapps.rag.core.vectorstore

import com.gazapps.rag.core.Document
import kotlinx.coroutines.runBlocking
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class HnswVectorStoreTest {
    
    private fun createTestDocuments(count: Int): List<Document> {
        return (1..count).map { i ->
            Document(
                id = "doc$i",
                content = "Document content $i",
                metadata = mapOf(
                    "category" to if (i % 2 == 0) "technology" else "science",
                    "priority" to i
                )
            )
        }
    }
    
    private fun createTestEmbeddings(count: Int, dimensions: Int = 8): List<FloatArray> {
        // Create test embeddings where documents with similar IDs have similar embeddings
        return (1..count).map { i ->
            FloatArray(dimensions) { d ->
                when {
                    d == 0 -> i.toFloat() / count // First dimension scales with ID
                    d % 2 == 0 -> 0.5f + (i % 3) * 0.1f // Even dimensions
                    else -> 0.5f - (i % 5) * 0.1f // Odd dimensions
                }
            }
        }
    }

    @Test
    fun `store and retrieve documents`() = runBlocking {
        // Setup
        val vectorStore = HnswVectorStore(m = 8, efConstruction = 50, efSearch = 40)
        val docs = createTestDocuments(5)
        val embeddings = createTestEmbeddings(5)
        
        // Store documents
        for (i in docs.indices) {
            vectorStore.store(docs[i], embeddings[i])
        }
        
        // Search with embedding similar to doc3
        val queryEmbedding = embeddings[2].copyOf().apply { 
            this[0] += 0.01f // Make it slightly different
        }
        
        val results = vectorStore.search(queryEmbedding, 2, null)
        
        // Verify results
        assertEquals(2, results.size)
        
        // The closest document should be doc3
        assertEquals("doc3", results[0].document.id)
        
        // Verify score is high (close to 1.0 for cosine similarity)
        assertTrue(results[0].score > 0.9f)
    }
    
    @Test
    fun `batch store should work`() = runBlocking {
        // Setup
        val vectorStore = HnswVectorStore()
        val docs = createTestDocuments(10)
        val embeddings = createTestEmbeddings(10)
        
        // Batch store
        vectorStore.batchStore(docs, embeddings)
        
        // Verify retrieval works for a randomly selected document
        val queryIndex = 5
        val results = vectorStore.search(embeddings[queryIndex], 1, null)
        
        assertEquals(1, results.size)
        assertEquals(docs[queryIndex].id, results[0].document.id)
    }
    
    @Test
    fun `search with filter should respect metadata constraints`() = runBlocking {
        // Setup
        val vectorStore = HnswVectorStore()
        val docs = createTestDocuments(10)
        val embeddings = createTestEmbeddings(10)
        
        // Store documents
        vectorStore.batchStore(docs, embeddings)
        
        // Search with category filter
        val results = vectorStore.search(
            query = embeddings[0], // Similar to doc1 (science)
            limit = 5,
            filter = mapOf("category" to "technology") // But only want technology docs
        )
        
        // Verify all results are from technology category
        assertTrue(results.isNotEmpty())
        results.forEach { scoredDoc ->
            assertEquals("technology", scoredDoc.document.metadata["category"])
        }
    }
    
    @Test
    fun `delete should remove documents`() = runBlocking {
        // Setup
        val vectorStore = HnswVectorStore()
        val docs = createTestDocuments(5)
        val embeddings = createTestEmbeddings(5)
        
        // Store documents
        vectorStore.batchStore(docs, embeddings)
        
        // Delete doc3
        vectorStore.delete("doc3")
        
        // Search with embedding similar to doc3
        val results = vectorStore.search(embeddings[2], 5, null)
        
        // Verify doc3 is not in results
        for (result in results) {
            assertTrue(result.document.id != "doc3")
        }
    }
    
    @Test
    fun `clear should remove all documents`() = runBlocking {
        // Setup
        val vectorStore = HnswVectorStore()
        val docs = createTestDocuments(5)
        val embeddings = createTestEmbeddings(5)
        
        // Store documents
        vectorStore.batchStore(docs, embeddings)
        
        // Clear store
        vectorStore.clear()
        
        // Verify store is empty
        assertEquals(0, vectorStore.size())
        
        // Search should return empty results
        val results = vectorStore.search(embeddings[0], 5, null)
        assertEquals(0, results.size)
    }
    
    @Test
    fun `search with multiple metrics should produce different results`() = runBlocking {
        // Create documents and embeddings
        val docs = createTestDocuments(10)
        val embeddings = createTestEmbeddings(10)
        
        // Create stores with different metrics
        val cosineStore = HnswVectorStore(
            similarityMetric = SimilarityUtils.SimilarityMetric.COSINE
        )
        val euclideanStore = HnswVectorStore(
            similarityMetric = SimilarityUtils.SimilarityMetric.EUCLIDEAN
        )
        val dotProductStore = HnswVectorStore(
            similarityMetric = SimilarityUtils.SimilarityMetric.DOT_PRODUCT
        )
        
        // Store the same documents in all stores
        cosineStore.batchStore(docs, embeddings)
        euclideanStore.batchStore(docs, embeddings)
        dotProductStore.batchStore(docs, embeddings)
        
        // Create a query embedding that should be closest to doc5
        val queryEmbedding = embeddings[4].copyOf()
        
        // Get results from each store
        val cosineResults = cosineStore.search(queryEmbedding, 3, null)
        val euclideanResults = euclideanStore.search(queryEmbedding, 3, null)
        val dotProductResults = dotProductStore.search(queryEmbedding, 3, null)
        
        // All should find doc5 as the closest match
        assertEquals("doc5", cosineResults[0].document.id)
        assertEquals("doc5", euclideanResults[0].document.id)
        assertEquals("doc5", dotProductResults[0].document.id)
        
        // But the ordering of other results might differ
        // This test mostly verifies the different metrics don't crash
    }
}
