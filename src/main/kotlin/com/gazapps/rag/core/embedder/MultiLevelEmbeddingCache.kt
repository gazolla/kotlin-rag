package com.gazapps.rag.core.embedder

import com.gazapps.rag.core.Embedder
import com.gazapps.rag.core.cache.Cache
import com.gazapps.rag.core.cache.DiskCache
import com.gazapps.rag.core.cache.MemoryCache
import com.gazapps.rag.core.cache.MultiLevelCache
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.slf4j.LoggerFactory
import java.io.File
import java.io.Serializable
import java.nio.ByteBuffer
import java.security.MessageDigest

/**
 * Multi-level caching implementation for embeddings
 * 
 * This combines in-memory caching for speed with disk caching for persistence.
 */
class MultiLevelEmbeddingCache(
    private val memorySize: Int = 1000,
    private val diskCacheDirectory: File? = null,
    private val diskCacheSize: Int = 10000,
    private val enableStats: Boolean = true
) : EmbeddingCache {
    private val logger = LoggerFactory.getLogger(MultiLevelEmbeddingCache::class.java)
    
    // Create the cache instance based on config
    private val cache: Cache<String, FloatArray> = if (diskCacheDirectory != null) {
        createMultiLevelCache()
    } else {
        createMemoryOnlyCache()
    }
    
    // Counters for cache stats
    private var cacheHits = 0
    private var cacheMisses = 0
    private var batchHits = 0
    private var batchMisses = 0
    
    /**
     * Get embedding from cache
     */
    override suspend fun get(text: String): FloatArray? {
        val key = createCacheKey(text)
        val result = cache.get(key)
        
        if (enableStats) {
            if (result != null) {
                cacheHits++
            } else {
                cacheMisses++
            }
        }
        
        return result
    }
    
    /**
     * Store embedding in cache
     */
    override suspend fun put(text: String, embedding: FloatArray) {
        val key = createCacheKey(text)
        cache.put(key, embedding)
    }
    
    /**
     * Get multiple embeddings from cache
     */
    override suspend fun batchGet(texts: List<String>): Map<String, FloatArray> {
        return withContext(Dispatchers.Default) {
            val results = mutableMapOf<String, FloatArray>()
            
            for (text in texts) {
                val embedding = get(text)
                if (embedding != null) {
                    results[text] = embedding
                    if (enableStats) {
                        batchHits++
                    }
                } else if (enableStats) {
                    batchMisses++
                }
            }
            
            results
        }
    }
    
    /**
     * Store multiple embeddings in cache
     */
    override suspend fun batchPut(textEmbeddings: Map<String, FloatArray>) {
        withContext(Dispatchers.Default) {
            for ((text, embedding) in textEmbeddings) {
                put(text, embedding)
            }
        }
    }
    
    /**
     * Clear the cache
     */
    override suspend fun clear() {
        cache.clear()
    }
    
    /**
     * Get the hit rate of the cache
     */
    override fun getStats(): EmbeddingCacheStats {
        val individualHitRate = if (cacheHits + cacheMisses > 0) {
            cacheHits.toFloat() / (cacheHits + cacheMisses)
        } else {
            0.0f
        }
        
        val batchHitRate = if (batchHits + batchMisses > 0) {
            batchHits.toFloat() / (batchHits + batchMisses)
        } else {
            0.0f
        }
        
        val cacheStats = cache.stats()
        
        return EmbeddingCacheStats(
            hits = cacheHits,
            misses = cacheMisses,
            batchHits = batchHits,
            batchMisses = batchMisses,
            hitRate = individualHitRate,
            batchHitRate = batchHitRate,
            size = cacheStats.size,
            capacity = cacheStats.capacity
        )
    }
    
    /**
     * Create a cache key for a text string
     */
    private fun createCacheKey(text: String): String {
        // Use SHA-256 for a secure, fixed-length hash
        val digest = MessageDigest.getInstance("SHA-256")
        val hashBytes = digest.digest(text.toByteArray())
        
        // Convert to hex string
        return hashBytes.joinToString("") { "%02x".format(it) }
    }
    
    /**
     * Create a multi-level cache with memory and disk layers
     */
    private fun createMultiLevelCache(): Cache<String, FloatArray> {
        // Ensure disk cache directory exists
        diskCacheDirectory?.mkdirs()
        
        val memoryCache = MemoryCache<String, FloatArray>(memorySize)
        
        val diskCache = DiskCache<String, FloatArray>(
            directory = diskCacheDirectory!!,
            serializer = { embedding -> serializeEmbedding(embedding) },
            deserializer = { bytes -> deserializeEmbedding(bytes) },
            keyToFilename = { key -> "$key.embedding" },
            maxEntries = diskCacheSize
        )
        
        return MultiLevelCache(
            listOf(
                memoryCache to 1, // Higher priority
                diskCache to 2     // Lower priority
            ),
            propagateWrites = true
        )
    }
    
    /**
     * Create a memory-only cache
     */
    private fun createMemoryOnlyCache(): Cache<String, FloatArray> {
        return MemoryCache(memorySize)
    }
    
    /**
     * Serialize a float array to bytes
     */
    private fun serializeEmbedding(embedding: FloatArray): ByteArray {
        val buffer = ByteBuffer.allocate(embedding.size * 4)
        for (value in embedding) {
            buffer.putFloat(value)
        }
        return buffer.array()
    }
    
    /**
     * Deserialize bytes to a float array
     */
    private fun deserializeEmbedding(bytes: ByteArray): FloatArray {
        val buffer = ByteBuffer.wrap(bytes)
        val result = FloatArray(bytes.size / 4)
        for (i in result.indices) {
            result[i] = buffer.getFloat()
        }
        return result
    }
}

/**
 * Embedder implementation that uses multi-level caching
 */
class CachedMultiLevelEmbedder(
    private val delegate: Embedder,
    memorySize: Int = 1000,
    diskCacheDirectory: File? = null,
    diskCacheSize: Int = 10000
) : Embedder {
    private val cache = MultiLevelEmbeddingCache(
        memorySize = memorySize,
        diskCacheDirectory = diskCacheDirectory,
        diskCacheSize = diskCacheSize
    )
    
    override suspend fun embed(text: String): FloatArray {
        // Try to get from cache
        val cached = cache.get(text)
        if (cached != null) {
            return cached
        }
        
        // Get from delegate if not in cache
        val embedding = delegate.embed(text)
        
        // Store in cache
        cache.put(text, embedding)
        
        return embedding
    }
    
    override suspend fun batchEmbed(texts: List<String>): List<FloatArray> {
        // Try to get as many as possible from cache
        val cachedEmbeddings = cache.batchGet(texts)
        
        // Collect texts that need to be embedded
        val missingTexts = texts.filter { it !in cachedEmbeddings }
        
        if (missingTexts.isEmpty()) {
            // All embeddings found in cache
            return texts.map { cachedEmbeddings[it]!! }
        }
        
        // Get missing embeddings from delegate
        val newEmbeddings = delegate.batchEmbed(missingTexts)
        
        // Store new embeddings in cache
        val newEmbeddingsMap = missingTexts.zip(newEmbeddings).toMap()
        cache.batchPut(newEmbeddingsMap)
        
        // Combine cached and new embeddings
        return texts.map { text ->
            cachedEmbeddings[text] ?: newEmbeddingsMap[text]!!
        }
    }
    
    /**
     * Get cache statistics
     */
    fun getCacheStats(): EmbeddingCacheStats {
        return cache.getStats()
    }
}
