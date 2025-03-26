package com.gazapps.rag.core.vectorstore

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class SimilarityUtilsTest {

    @Test
    fun `cosineSimilarity should return 1 for identical vectors`() {
        val vector1 = floatArrayOf(1f, 2f, 3f)
        val vector2 = floatArrayOf(1f, 2f, 3f)
        
        val similarity = SimilarityUtils.cosineSimilarity(vector1, vector2)
        
        assertEquals(1f, similarity, 0.0001f)
    }
    
    @Test
    fun `cosineSimilarity should return close to 0 for orthogonal vectors`() {
        val vector1 = floatArrayOf(1f, 0f, 0f)
        val vector2 = floatArrayOf(0f, 1f, 0f)
        
        val similarity = SimilarityUtils.cosineSimilarity(vector1, vector2)
        
        assertEquals(0f, similarity, 0.0001f)
    }
    
    @Test
    fun `cosineSimilarity should return -1 for opposite vectors`() {
        val vector1 = floatArrayOf(1f, 2f, 3f)
        val vector2 = floatArrayOf(-1f, -2f, -3f)
        
        val similarity = SimilarityUtils.cosineSimilarity(vector1, vector2)
        
        assertEquals(-1f, similarity, 0.0001f)
    }
    
    @Test
    fun `dotProduct should return correct value`() {
        val vector1 = floatArrayOf(1f, 2f, 3f)
        val vector2 = floatArrayOf(4f, 5f, 6f)
        
        val dotProduct = SimilarityUtils.dotProduct(vector1, vector2)
        
        // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        assertEquals(32f, dotProduct, 0.0001f)
    }
    
    @Test
    fun `euclideanSimilarity should return 1 for identical vectors`() {
        val vector1 = floatArrayOf(1f, 2f, 3f)
        val vector2 = floatArrayOf(1f, 2f, 3f)
        
        val similarity = SimilarityUtils.euclideanSimilarity(vector1, vector2)
        
        assertEquals(1f, similarity, 0.0001f)
    }
    
    @Test
    fun `euclideanSimilarity should decrease as distance increases`() {
        val vector1 = floatArrayOf(0f, 0f, 0f)
        val vector2 = floatArrayOf(1f, 1f, 1f)
        val vector3 = floatArrayOf(2f, 2f, 2f)
        
        val similarity1 = SimilarityUtils.euclideanSimilarity(vector1, vector2)
        val similarity2 = SimilarityUtils.euclideanSimilarity(vector1, vector3)
        
        assertTrue(similarity1 > similarity2)
    }
    
    @Test
    fun `manhattanSimilarity should return 1 for identical vectors`() {
        val vector1 = floatArrayOf(1f, 2f, 3f)
        val vector2 = floatArrayOf(1f, 2f, 3f)
        
        val similarity = SimilarityUtils.manhattanSimilarity(vector1, vector2)
        
        assertEquals(1f, similarity, 0.0001f)
    }
    
    @Test
    fun `angularSimilarity should return 1 for identical vectors`() {
        val vector1 = floatArrayOf(1f, 2f, 3f)
        val vector2 = floatArrayOf(1f, 2f, 3f)
        
        val similarity = SimilarityUtils.angularSimilarity(vector1, vector2)
        
        assertEquals(1f, similarity, 0.0001f)
    }
    
    @Test
    fun `rbfSimilarity should return 1 for identical vectors`() {
        val vector1 = floatArrayOf(1f, 2f, 3f)
        val vector2 = floatArrayOf(1f, 2f, 3f)
        
        val similarity = SimilarityUtils.rbfSimilarity(vector1, vector2)
        
        assertEquals(1f, similarity, 0.0001f)
    }
    
    @Test
    fun `normalize should produce unit vectors`() {
        val vector = floatArrayOf(3f, 4f)
        
        val normalized = SimilarityUtils.normalize(vector)
        
        // Length should be 1.0
        val length = Math.sqrt((normalized[0] * normalized[0] + normalized[1] * normalized[1]).toDouble())
        assertEquals(1.0, length, 0.0001)
    }
    
    @Test
    fun `calculateSimilarity should use the specified metric`() {
        val vector1 = floatArrayOf(1f, 0f)
        val vector2 = floatArrayOf(0f, 1f)
        
        val cosineSimilarity = SimilarityUtils.calculateSimilarity(
            vector1, vector2, SimilarityUtils.SimilarityMetric.COSINE
        )
        
        val euclideanSimilarity = SimilarityUtils.calculateSimilarity(
            vector1, vector2, SimilarityUtils.SimilarityMetric.EUCLIDEAN
        )
        
        assertEquals(0f, cosineSimilarity, 0.0001f)
        assertTrue(euclideanSimilarity > 0f)
    }
    
    @Test
    fun `isNormalized should return true for normalized vectors`() {
        val vector = floatArrayOf(3f, 4f)
        val normalized = SimilarityUtils.normalize(vector)
        
        val result = SimilarityUtils.isNormalized(normalized)
        
        assertTrue(result)
    }
}
