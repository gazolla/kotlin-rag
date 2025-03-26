package com.gazapps.rag.core.reranking

import com.gazapps.rag.core.Document
import com.gazapps.rag.core.SimpleDocument
import com.gazapps.rag.core.ScoredDocument
import org.junit.jupiter.api.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotEquals
import kotlin.test.assertTrue

class RerankerTest {

    @Test
    fun `MMR reranker should balance relevance and diversity`() {
        // Create similar test documents
        val docs = listOf(
            createScoredDocument("doc1", "Kotlin is a modern programming language", 0.9f),
            createScoredDocument("doc2", "Kotlin is a programming language for the JVM", 0.85f),
            createScoredDocument("doc3", "Java is another programming language for the JVM", 0.8f),
            createScoredDocument("doc4", "Python is a programming language", 0.75f),
            createScoredDocument("doc5", "Swift is Apple's programming language", 0.7f)
        )
        
        // Create query embedding
        val queryEmbedding = FloatArray(5) { 0.2f }
        
        // Test with diversity weight = 0 (only relevance)
        val rerankerRelevanceOnly = MaximalMarginalRelevanceReranker()
        val resultRelevanceOnly = rerankerRelevanceOnly.rerank(
            queryEmbedding,
            docs,
            RerankingOptions(diversityWeight = 0.0f)
        )
        
        // The first result should be the most relevant
        assertEquals("doc1", resultRelevanceOnly.first().document.id)
        
        // Test with diversity weight = 0.7 (balance relevance and diversity)
        val rerankerBalanced = MaximalMarginalRelevanceReranker()
        val resultBalanced = rerankerBalanced.rerank(
            queryEmbedding,
            docs,
            RerankingOptions(diversityWeight = 0.7f)
        )
        
        // The second result with diversity should be different than without diversity
        assertNotEquals(
            resultRelevanceOnly[1].document.id,
            resultBalanced[1].document.id
        )
        
        // Documents about non-JVM languages should be ranked higher with diversity
        val pythonOrSwiftInTopThree = resultBalanced.take(3)
            .any { it.document.id == "doc4" || it.document.id == "doc5" }
        
        assertTrue(pythonOrSwiftInTopThree)
    }
    
    @Test
    fun `Ensemble reranker should combine scores from multiple rerankers`() {
        // Create test documents
        val docs = listOf(
            createScoredDocument("doc1", "Kotlin is a modern programming language", 0.9f),
            createScoredDocument("doc2", "Java is a programming language for the JVM", 0.7f),
            createScoredDocument("doc3", "Python is a programming language", 0.5f)
        )
        
        // Create query embedding
        val queryEmbedding = FloatArray(5) { 0.2f }
        
        // Create mock rerankers that will reorder the documents in different ways
        val reranker1 = object : Reranker {
            override fun rerank(
                queryEmbedding: FloatArray,
                documents: List<ScoredDocument>,
                options: RerankingOptions
            ): List<ScoredDocument> {
                // Reorder to doc3, doc1, doc2
                return listOf(
                    ScoredDocument(docs[2].document, 0.9f),
                    ScoredDocument(docs[0].document, 0.8f),
                    ScoredDocument(docs[1].document, 0.7f)
                )
            }
        }
        
        val reranker2 = object : Reranker {
            override fun rerank(
                queryEmbedding: FloatArray,
                documents: List<ScoredDocument>,
                options: RerankingOptions
            ): List<ScoredDocument> {
                // Reorder to doc2, doc1, doc3
                return listOf(
                    ScoredDocument(docs[1].document, 0.9f),
                    ScoredDocument(docs[0].document, 0.8f),
                    ScoredDocument(docs[2].document, 0.5f)
                )
            }
        }
        
        // Create ensemble with equal weights
        val ensembleReranker = EnsembleReranker(listOf(
            reranker1 to 1.0f,
            reranker2 to 1.0f
        ))
        
        val result = ensembleReranker.rerank(
            queryEmbedding,
            docs,
            RerankingOptions(limit = 3)
        )
        
        // The order should reflect the ensemble scores
        // doc1 should be first (0.8 from both)
        // doc2 and doc3 should follow (0.7+0.9 and 0.9+0.5)
        assertEquals("doc1", result[0].document.id)
    }
    
    private fun createScoredDocument(id: String, content: String, score: Float): ScoredDocument {
        val document = SimpleDocument(id, content, emptyMap())
        return ScoredDocument(document, score)
    }
}
