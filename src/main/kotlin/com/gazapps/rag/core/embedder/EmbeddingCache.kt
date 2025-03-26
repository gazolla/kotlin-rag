package com.gazapps.rag.core.embedder

/**
 * Cache for storing and retrieving embeddings
 */
interface EmbeddingCache {
    /**
     * Get an embedding from the cache
     *
     * @param text The text whose embedding to retrieve
     * @return The embedding or null if not in cache
     */
    suspend fun get(text: String): FloatArray?
    
    /**
     * Store an embedding in the cache
     *
     * @param text The text being embedded
     * @param embedding The embedding to store
     */
    suspend fun put(text: String, embedding: FloatArray)
    
    /**
     * Get multiple embeddings from the cache
     *
     * @param texts List of texts to get embeddings for
     * @return Map of text to embedding for cache hits
     */
    suspend fun batchGet(texts: List<String>): Map<String, FloatArray>
    
    /**
     * Store multiple embeddings in the cache
     *
     * @param textEmbeddings Map of text to embeddings
     */
    suspend fun batchPut(textEmbeddings: Map<String, FloatArray>)
    
    /**
     * Clear the cache
     */
    suspend fun clear()
    
    /**
     * Get cache statistics
     */
    fun getStats(): EmbeddingCacheStats
}

/**
 * Statistics for embedding cache
 */
data class EmbeddingCacheStats(
    val hits: Int,
    val misses: Int,
    val batchHits: Int,
    val batchMisses: Int,
    val hitRate: Float,
    val batchHitRate: Float,
    val size: Int,
    val capacity: Int
)
