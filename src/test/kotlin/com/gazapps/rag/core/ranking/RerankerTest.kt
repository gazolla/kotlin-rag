package com.gazapps.rag.core.ranking

import com.gazapps.rag.core.Document
import com.gazapps.rag.core.ScoredDocument
import com.gazapps.rag.core.SimpleDocument
import com.gazapps.rag.core.VectorUtils
import org.junit.jupiter.api.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotEquals
import kotlin.test.assertTrue

class RerankerTest {
    
    @Test
    fun `MMRReranker should balance relevance and diversity`() {
        // Create sample documents with embeddings
        val documents = createSampleDocuments()
        val queryEmbedding = floatArrayOf(1.0f, 0.0f, 0.0f) // Query closer to doc1
        
        // Create scored documents
        val scoredDocuments = documents.map { doc ->
            val embedding = doc.metadata["embedding"] as FloatArray
            val score = VectorUtils.cosineSimilarity(queryEmbedding, embedding)
            ScoredDocument(doc, score)
        }.sortedByDescending { it.score }
        
        // Check initial ordering based on similarity
        assertEquals("doc1", scoredDocuments[0].document.id, "Most similar should be doc1")
        assertEquals("doc2", scoredDocuments[1].document.id, "Second most similar should be doc2")
        assertEquals("doc3", scoredDocuments[2].document.id, "Third most similar should be doc3")
        
        // Rerank with MMR (high lambda = favor relevance)
        val mmrReranker = MMRReranker(lambda = 0.9f)
        val mmrResults = mmrReranker.rerank(queryEmbedding, scoredDocuments)
        
        // With high lambda, order should be similar to original
        assertEquals("doc1", mmrResults[0].document.id, "First result should still be doc1")
        
        // Rerank with MMR (low lambda = favor diversity)
        val diverseReranker = MMRReranker(lambda = 0.1f)
        val diverseResults = diverseReranker.rerank(queryEmbedding, scoredDocuments)
        
        // With low lambda, order should prioritize diversity
        assertEquals("doc1", diverseResults[0].document.id, "First result should still be doc1 (most relevant)")
        
        // Due to doc2 and doc3 being similar to each other,
        // the second result should be the document that's most different from doc1
        assertEquals("doc4", diverseResults[1].document.id, "Second result should be doc4 for diversity")
    }
    
    @Test
    fun `MetadataBoostReranker should boost documents with specified metadata`() {
        // Create documents with different metadata
        val doc1 = SimpleDocument("doc1", "Content 1", mapOf("important" to true, "score" to 5))
        val doc2 = SimpleDocument("doc2", "Content 2", mapOf("important" to false, "score" to 3))
        val doc3 = SimpleDocument("doc3", "Content 3", mapOf("score" to 4))
        
        val scoredDocuments = listOf(
            ScoredDocument(doc1, 0.7f),
            ScoredDocument(doc2, 0.8f), // Initially highest score
            ScoredDocument(doc3, 0.6f)
        )
        
        // Boost documents with "important" metadata
        val boostReranker = MetadataBoostReranker(mapOf("important" to 1.5f))
        val reranked = boostReranker.rerank(floatArrayOf(1.0f), scoredDocuments)
        
        // Doc1 should now be first due to the boost
        assertEquals("doc1", reranked[0].document.id, "Doc1 should be first after boosting")
        assertTrue(reranked[0].score > reranked[1].score, "Doc1 score should be higher after boosting")
    }
    
    @Test
    fun `CompositeReranker should apply multiple rerankers in sequence`() {
        // Create documents
        val docs = createSampleDocuments()
        val queryEmbedding = floatArrayOf(1.0f, 0.0f, 0.0f)
        
        // Create initial scored documents
        val scoredDocs = docs.map { doc ->
            val embedding = doc.metadata["embedding"] as FloatArray
            val score = VectorUtils.cosineSimilarity(queryEmbedding, embedding)
            ScoredDocument(doc, score)
        }.sortedByDescending { it.score }
        
        // First reranker: metadata boost for "recent" documents
        val boostReranker = MetadataBoostReranker(mapOf("recent" to 1.5f))
        
        // Second reranker: MMR for diversity
        val mmrReranker = MMRReranker(lambda = 0.5f)
        
        // Composite reranker
        val compositeReranker = CompositeReranker.of(boostReranker, mmrReranker)
        
        // Apply composite reranking
        val reranked = compositeReranker.rerank(queryEmbedding, scoredDocs)
        
        // Check that both rerankers had an effect
        assertNotEquals(
            scoredDocs.map { it.document.id },
            reranked.map { it.document.id },
            "Composite reranker should change document order"
        )
        
        // Doc4 is recent and should be boosted higher than its original position
        val doc4OriginalPosition = scoredDocs.indexOfFirst { it.document.id == "doc4" }
        val doc4NewPosition = reranked.indexOfFirst { it.document.id == "doc4" }
        assertTrue(doc4NewPosition < doc4OriginalPosition, "Doc4 should be ranked higher after boosting")
    }
    
    // Helper function to create sample documents with embeddings
    private fun createSampleDocuments(): List<Document> {
        return listOf(
            SimpleDocument(
                id = "doc1",
                content = "This is the first document about Kotlin",
                metadata = mapOf(
                    "embedding" to floatArrayOf(0.9f, 0.1f, 0.0f),
                    "recent" to false
                )
            ),
            SimpleDocument(
                id = "doc2",
                content = "This is the second document about Java",
                metadata = mapOf(
                    "embedding" to floatArrayOf(0.7f, 0.3f, 0.1f),
                    "recent" to false
                )
            ),
            SimpleDocument(
                id = "doc3",
                content = "This is the third document about programming languages",
                metadata = mapOf(
                    "embedding" to floatArrayOf(0.6f, 0.4f, 0.2f),
                    "recent" to false
                )
            ),
            SimpleDocument(
                id = "doc4",
                content = "This is a document about something completely different",
                metadata = mapOf(
                    "embedding" to floatArrayOf(0.1f, 0.2f, 0.9f),
                    "recent" to true
                )
            )
        )
    }
}
