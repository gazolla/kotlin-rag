package com.gazapps.rag.core

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import kotlin.math.abs

class VectorUtilsTest {
    
    @Test
    fun `cosineSimilarity should calculate correct similarity`() {
        // Setup
        val v1 = floatArrayOf(1f, 0f, 0f)
        val v2 = floatArrayOf(1f, 0f, 0f)
        val v3 = floatArrayOf(0f, 1f, 0f)
        val v4 = floatArrayOf(0.7071f, 0.7071f, 0f) // 45 degrees between v1 and v3
        
        // Act & Assert
        assertEquals(1.0f, VectorUtils.cosineSimilarity(v1, v2), 0.0001f)
        assertEquals(0.0f, VectorUtils.cosineSimilarity(v1, v3), 0.0001f)
        assertApproximatelyEqual(0.7071f, VectorUtils.cosineSimilarity(v1, v4), 0.001f)
    }
    
    @Test
    fun `dotProduct should calculate correct similarity`() {
        // Setup
        val v1 = floatArrayOf(1f, 2f, 3f)
        val v2 = floatArrayOf(4f, 5f, 6f)
        
        // Act
        val result = VectorUtils.dotProduct(v1, v2)
        
        // Assert
        assertEquals(32.0f, result, 0.0001f) // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    }
    
    @Test
    fun `euclideanDistance should calculate correct distance`() {
        // Setup
        val v1 = floatArrayOf(0f, 0f, 0f)
        val v2 = floatArrayOf(3f, 4f, 0f)
        
        // Act
        val result = VectorUtils.euclideanDistance(v1, v2)
        
        // Assert
        assertEquals(5.0f, result, 0.0001f) // sqrt(3^2 + 4^2) = sqrt(9 + 16) = sqrt(25) = 5
    }
    
    @Test
    fun `normalize should make vector unit length`() {
        // Setup
        val vector = floatArrayOf(3f, 4f, 0f)
        
        // Act
        VectorUtils.normalize(vector)
        
        // Assert
        assertEquals(0.6f, vector[0], 0.0001f) // 3/5
        assertEquals(0.8f, vector[1], 0.0001f) // 4/5
        assertEquals(0.0f, vector[2], 0.0001f)
        
        // Magnitude should be 1.0
        val magnitude = vector.fold(0f) { acc, value -> acc + value * value }
        assertApproximatelyEqual(1.0f, magnitude, 0.0001f)
    }
    
    @Test
    fun `normalized should return new normalized vector`() {
        // Setup
        val original = floatArrayOf(3f, 4f, 0f)
        
        // Act
        val normalized = VectorUtils.normalized(original)
        
        // Assert
        // Original should be unchanged
        assertEquals(3.0f, original[0], 0.0001f)
        assertEquals(4.0f, original[1], 0.0001f)
        assertEquals(0.0f, original[2], 0.0001f)
        
        // Normalized should be unit length
        assertEquals(0.6f, normalized[0], 0.0001f) // 3/5
        assertEquals(0.8f, normalized[1], 0.0001f) // 4/5
        assertEquals(0.0f, normalized[2], 0.0001f)
    }
    
    @Test
    fun `normalize should handle zero vectors`() {
        // Setup
        val zeroVector = floatArrayOf(0f, 0f, 0f)
        
        // Act
        VectorUtils.normalize(zeroVector)
        
        // Assert
        assertEquals(0.0f, zeroVector[0], 0.0001f)
        assertEquals(0.0f, zeroVector[1], 0.0001f)
        assertEquals(0.0f, zeroVector[2], 0.0001f)
    }
    
    @Test
    fun `euclideanToSimilarity should convert distance to similarity`() {
        // Act & Assert
        assertEquals(1.0f, VectorUtils.euclideanToSimilarity(0f), 0.0001f) // Same vector
        assertEquals(0.5f, VectorUtils.euclideanToSimilarity(1f), 0.0001f) // Distance of 1
        assertEquals(0.2f, VectorUtils.euclideanToSimilarity(4f), 0.0001f) // Distance of 4
        
        // As distance approaches infinity, similarity should approach 0
        val farSimilarity = VectorUtils.euclideanToSimilarity(1000f)
        assertTrue(farSimilarity < 0.01f)
    }
    
    private fun assertApproximatelyEqual(expected: Float, actual: Float, delta: Float) {
        assertTrue(abs(expected - actual) <= delta,
                 "Expected $expected but got $actual (allowed difference: $delta)")
    }
}
