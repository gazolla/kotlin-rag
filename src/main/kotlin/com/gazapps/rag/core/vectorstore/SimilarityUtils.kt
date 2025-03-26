package com.gazapps.rag.core.vectorstore

import kotlin.math.exp
import kotlin.math.pow
import kotlin.math.sqrt

/**
 * Utility class for vector similarity calculations and operations
 */
object SimilarityUtils {
    /**
     * Available similarity metrics
     */
    enum class SimilarityMetric {
        COSINE,         // Standard cosine similarity (angle between vectors)
        DOT_PRODUCT,    // Simple dot product (magnitude sensitive)
        EUCLIDEAN,      // Euclidean distance converted to similarity
        ANGULAR,        // Angular distance-based similarity
        MANHATTAN,      // Manhattan distance converted to similarity
        RBF             // Radial Basis Function kernel (Gaussian)
    }
    
    /**
     * Calculate cosine similarity between two vectors
     */
    fun cosineSimilarity(a: FloatArray, b: FloatArray): Float {
        require(a.size == b.size) { "Vector dimensions don't match: ${a.size} vs ${b.size}" }
        
        var dotProduct = 0f
        var normA = 0f
        var normB = 0f
        
        for (i in a.indices) {
            dotProduct += a[i] * b[i]
            normA += a[i] * a[i]
            normB += b[i] * b[i]
        }
        
        return if (normA > 0 && normB > 0) {
            dotProduct / (sqrt(normA) * sqrt(normB))
        } else {
            0f
        }
    }
    
    /**
     * Calculate dot product similarity between two vectors
     */
    fun dotProduct(a: FloatArray, b: FloatArray): Float {
        require(a.size == b.size) { "Vector dimensions don't match: ${a.size} vs ${b.size}" }
        
        var sum = 0f
        for (i in a.indices) {
            sum += a[i] * b[i]
        }
        return sum
    }
    
    /**
     * Calculate Euclidean distance-based similarity between two vectors
     * Maps distance to similarity with 1/(1+distance)
     */
    fun euclideanSimilarity(a: FloatArray, b: FloatArray): Float {
        require(a.size == b.size) { "Vector dimensions don't match: ${a.size} vs ${b.size}" }
        
        var sum = 0f
        for (i in a.indices) {
            val diff = a[i] - b[i]
            sum += diff * diff
        }
        val distance = sqrt(sum)
        
        // Convert distance to similarity (1 / (1 + distance))
        return 1f / (1f + distance)
    }
    
    /**
     * Calculate Manhattan distance-based similarity between two vectors
     * Maps distance to similarity with 1/(1+distance)
     */
    fun manhattanSimilarity(a: FloatArray, b: FloatArray): Float {
        require(a.size == b.size) { "Vector dimensions don't match: ${a.size} vs ${b.size}" }
        
        var distance = 0f
        for (i in a.indices) {
            distance += kotlin.math.abs(a[i] - b[i])
        }
        
        // Convert distance to similarity (1 / (1 + distance))
        return 1f / (1f + distance)
    }
    
    /**
     * Calculate angular similarity between two vectors
     * Similar to cosine but scales differently
     */
    fun angularSimilarity(a: FloatArray, b: FloatArray): Float {
        val cosine = cosineSimilarity(a, b)
        // Convert cosine to angular similarity
        return (1f - (kotlin.math.acos(cosine.coerceIn(-1f, 1f)) / kotlin.math.PI)).toFloat()
    }
    
    /**
     * Calculate Radial Basis Function kernel similarity (Gaussian)
     * Good for clusters and non-linear relationships
     */
    fun rbfSimilarity(a: FloatArray, b: FloatArray, gamma: Float = 1f): Float {
        require(a.size == b.size) { "Vector dimensions don't match: ${a.size} vs ${b.size}" }
        
        var sum = 0f
        for (i in a.indices) {
            val diff = a[i] - b[i]
            sum += diff * diff
        }
        
        return exp(-gamma * sum).toFloat()
    }
    
    /**
     * Calculate similarity between two vectors using specified metric
     */
    fun calculateSimilarity(
        a: FloatArray, 
        b: FloatArray, 
        metric: SimilarityMetric = SimilarityMetric.COSINE
    ): Float {
        return when (metric) {
            SimilarityMetric.COSINE -> cosineSimilarity(a, b)
            SimilarityMetric.DOT_PRODUCT -> dotProduct(a, b)
            SimilarityMetric.EUCLIDEAN -> euclideanSimilarity(a, b)
            SimilarityMetric.ANGULAR -> angularSimilarity(a, b)
            SimilarityMetric.MANHATTAN -> manhattanSimilarity(a, b)
            SimilarityMetric.RBF -> rbfSimilarity(a, b)
        }
    }
    
    /**
     * Normalize a vector to unit length
     */
    fun normalize(vector: FloatArray): FloatArray {
        var sum = 0f
        for (value in vector) {
            sum += value * value
        }
        
        val norm = sqrt(sum)
        
        return if (norm > 0) {
            FloatArray(vector.size) { i -> vector[i] / norm }
        } else {
            vector.clone()
        }
    }
    
    /**
     * L2 normalization for a list of vectors
     */
    fun batchNormalize(vectors: List<FloatArray>): List<FloatArray> {
        return vectors.map { normalize(it) }
    }
    
    /**
     * Enhanced cosine similarity for two normalized vectors (optimization)
     * Assumes both vectors are already normalized
     */
    fun normalizedCosineSimilarity(a: FloatArray, b: FloatArray): Float {
        require(a.size == b.size) { "Vector dimensions don't match: ${a.size} vs ${b.size}" }
        
        var dotProduct = 0f
        for (i in a.indices) {
            dotProduct += a[i] * b[i]
        }
        
        return dotProduct
    }
    
    /**
     * Check if a vector is approximately normalized
     */
    fun isNormalized(vector: FloatArray, tolerance: Float = 1e-4f): Boolean {
        var sum = 0f
        for (value in vector) {
            sum += value * value
        }
        
        return kotlin.math.abs(1f - sqrt(sum)) <= tolerance
    }
}
