package com.gazapps.rag.core.vectorstore

import com.gazapps.rag.core.Document
import com.gazapps.rag.core.SimpleDocument
import com.gazapps.rag.core.ScoredDocument
import com.gazapps.rag.core.VectorStore
import com.gazapps.rag.core.vectorstore.SimilarityUtils
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import org.slf4j.LoggerFactory
import redis.clients.jedis.JedisPool
import redis.clients.jedis.JedisPoolConfig
import java.util.concurrent.ConcurrentHashMap
import kotlin.ranges.ClosedRange

/**
 * VectorStore implementation using Redis.
 * 
 * NOTE: This implementation requires Redis Stack with RedisJSON and RediSearch modules.
 * Full implementation would use advanced Redis features which require additional dependencies.
 * 
 * This class provides a basic implementation for development purposes.
 * For production use, you'll need the complete Redis stack with vector search capabilities.
 *
 * @param host Redis host
 * @param port Redis port
 * @param password Redis password (optional)
 * @param keyPrefix Prefix for Redis keys
 * @param dimensions Number of dimensions in the embedding vectors
 */
class RedisVectorStore(
    host: String = "localhost",
    port: Int = 6379,
    password: String? = null,
    private val keyPrefix: String = "doc:",
    private val dimensions: Int = 1536,
    private val distance: String = "COSINE"
) : VectorStore {
    
    private val logger = LoggerFactory.getLogger(RedisVectorStore::class.java)
    
    private val json = Json { 
        ignoreUnknownKeys = true 
        prettyPrint = false
    }
    
    // In-memory storage for development/testing purposes
    private val documentStore = ConcurrentHashMap<String, Document>()
    private val embeddingStore = ConcurrentHashMap<String, FloatArray>()
    
    private val pool: JedisPool
    
    init {
        logger.warn("This is a simplified RedisVectorStore implementation. For production use, add Redis Stack dependencies.")
        
        val poolConfig = JedisPoolConfig().apply {
            maxTotal = 32
            maxIdle = 16
            minIdle = 8
            testOnBorrow = true
            testOnReturn = true
            testWhileIdle = true
        }
        
        pool = if (password.isNullOrEmpty()) {
            JedisPool(poolConfig, host, port)
        } else {
            JedisPool(poolConfig, host, port, null, password)
        }
    }
    
    override suspend fun store(document: Document, embedding: FloatArray) = withContext(Dispatchers.IO) {
        try {
            val key = "$keyPrefix${document.id}"
            
            // Store in Redis as a hash
            pool.resource.use { jedis ->
                val docData = mapOf(
                    "id" to document.id,
                    "content" to document.content,
                    "metadata" to json.encodeToString(document.metadata)
                )
                
                jedis.hset(key, docData)
                
                // Store embedding in our in-memory map (would be vector in Redis Stack)
                embeddingStore[document.id] = embedding
                documentStore[document.id] = document
            }
            
            logger.debug("Stored document with ID: ${document.id}")
        } catch (e: Exception) {
            logger.error("Error storing document: ${e.message}", e)
            throw RuntimeException("Failed to store document in Redis", e)
        }
    }
    
    override suspend fun batchStore(documents: List<Document>, embeddings: List<FloatArray>) = withContext(Dispatchers.IO) {
        require(documents.size == embeddings.size) {
            "Number of documents (${documents.size}) must match number of embeddings (${embeddings.size})"
        }
        
        if (documents.isEmpty()) return@withContext
        
        try {
            pool.resource.use { jedis ->
                val pipeline = jedis.pipelined()
                
                for (i in documents.indices) {
                    val document = documents[i]
                    val embedding = embeddings[i]
                    val key = "$keyPrefix${document.id}"
                    
                    val docData = mapOf(
                        "id" to document.id,
                        "content" to document.content,
                        "metadata" to json.encodeToString(document.metadata)
                    )
                    
                    pipeline.hset(key, docData)
                    
                    // Store in memory
                    embeddingStore[document.id] = embedding
                    documentStore[document.id] = document
                }
                
                pipeline.sync()
                logger.debug("Batch stored ${documents.size} documents")
            }
        } catch (e: Exception) {
            logger.error("Error batch storing documents: ${e.message}", e)
            throw RuntimeException("Failed to batch store documents in Redis", e)
        }
    }
    
    override suspend fun search(
        query: FloatArray, 
        limit: Int, 
        filter: Map<String, Any>?
    ): List<ScoredDocument> = withContext(Dispatchers.IO) {
        logger.debug("Searching for similar documents to query vector of size ${query.size}")
        
        // In-memory implementation for development
        val results = embeddingStore.entries
            .asSequence()
            .map { (id, emb) -> 
                val doc = documentStore[id] ?: return@map null
                
                // Apply filter if provided
                if (filter != null && !MetadataFilter.matchesSimpleFilter(doc, filter)) {
                    return@map null
                }
                
                // Calculate similarity
                val similarity = SimilarityUtils.cosineSimilarity(query, emb)
                ScoredDocument(doc, similarity)
            }
            .filterNotNull()
            .sortedByDescending { it.score }
            .take(limit)
            .toList()
        
        logger.debug("Found ${results.size} matching documents")
        return@withContext results
    }
    
    override suspend fun delete(documentId: String) = withContext(Dispatchers.IO) {
        try {
            val key = "$keyPrefix$documentId"
            
            pool.resource.use { jedis ->
                jedis.del(key)
            }
            
            // Remove from in-memory stores
            embeddingStore.remove(documentId)
            documentStore.remove(documentId)
            
            logger.debug("Deleted document with ID: $documentId")
        } catch (e: Exception) {
            logger.error("Error deleting document: ${e.message}", e)
            throw RuntimeException("Failed to delete document from Redis", e)
        }
    }
    
    override suspend fun clear() = withContext(Dispatchers.IO) {
        try {
            // Delete all keys with our prefix
            pool.resource.use { jedis ->
                val keys = jedis.keys("$keyPrefix*")
                if (keys.isNotEmpty()) {
                    jedis.del(*keys.toTypedArray())
                }
            }
            
            // Clear in-memory stores
            embeddingStore.clear()
            documentStore.clear()
            
            logger.info("Cleared all documents with prefix: $keyPrefix")
        } catch (e: Exception) {
            logger.error("Error clearing documents: ${e.message}", e)
            throw RuntimeException("Failed to clear documents from Redis", e)
        }
    }
    
    /**
     * Close the Redis connection pool
     */
    fun close() {
        pool.close()
    }
    
    /**
     * Get the number of documents in the store
     */
    suspend fun count(): Long = withContext(Dispatchers.IO) {
        try {
            pool.resource.use { jedis ->
                return@withContext jedis.keys("$keyPrefix*").size.toLong()
            }
        } catch (e: Exception) {
            logger.error("Error getting document count: ${e.message}", e)
            throw RuntimeException("Failed to get document count from Redis", e)
        }
    }
}
