package com.gazapps.rag.core.cache

import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import org.slf4j.LoggerFactory
import java.io.File
import java.nio.ByteBuffer
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.ConcurrentLinkedQueue
import java.util.concurrent.atomic.AtomicInteger
import java.io.ObjectOutputStream
import java.io.ObjectInputStream
import java.io.ByteArrayOutputStream
import java.io.ByteArrayInputStream
import java.io.Serializable

/**
 * Generic cache interface
 */
interface Cache<K, V> {
    /**
     * Get a value from the cache
     *
     * @param key The key to look up
     * @return The cached value, or null if not found
     */
    suspend fun get(key: K): V?
    
    /**
     * Put a value into the cache
     *
     * @param key The key for the value
     * @param value The value to cache
     */
    suspend fun put(key: K, value: V)
    
    /**
     * Remove a value from the cache
     *
     * @param key The key to remove
     * @return true if the key was removed, false if it wasn't in the cache
     */
    suspend fun remove(key: K): Boolean
    
    /**
     * Clear all values from the cache
     */
    suspend fun clear()
    
    /**
     * Get the current size of the cache
     *
     * @return The number of entries in the cache
     */
    fun size(): Int
    
    /**
     * Get the hit rate of the cache
     *
     * @return The cache hit rate as a value between 0.0 and 1.0
     */
    fun stats(): CacheStats
}

/**
 * Cache statistics
 */
data class CacheStats(
    val hits: Int,
    val misses: Int,
    val size: Int,
    val capacity: Int
) {
    val hitRate: Double
        get() = if (hits + misses > 0) hits.toDouble() / (hits + misses) else 0.0
}

/**
 * In-memory LRU cache implementation
 *
 * @param maxSize Maximum number of entries in the cache
 */
class MemoryCache<K, V>(
    private val maxSize: Int = 1000
) : Cache<K, V> {
    private val logger = LoggerFactory.getLogger(MemoryCache::class.java)
    
    private val cache = ConcurrentHashMap<K, V>()
    private val accessQueue = ConcurrentLinkedQueue<K>()
    private val hits = AtomicInteger(0)
    private val misses = AtomicInteger(0)
    private val mutex = Mutex()
    
    override suspend fun get(key: K): V? {
        val result = cache[key]
        
        if (result != null) {
            updateAccessOrder(key)
            hits.incrementAndGet()
            logger.debug("Cache hit for key: {}", key)
        } else {
            misses.incrementAndGet()
            logger.debug("Cache miss for key: {}", key)
        }
        
        return result
    }
    
    override suspend fun put(key: K, value: V) {
        mutex.withLock {
            // Evict oldest entry if at capacity
            if (cache.size >= maxSize && !cache.containsKey(key)) {
                evictOldestEntry()
            }
            
            cache[key] = value
            updateAccessOrder(key)
            logger.debug("Added to cache: {}", key)
        }
    }
    
    override suspend fun remove(key: K): Boolean {
        mutex.withLock {
            accessQueue.remove(key)
            return cache.remove(key) != null
        }
    }
    
    override suspend fun clear() {
        mutex.withLock {
            cache.clear()
            accessQueue.clear()
            hits.set(0)
            misses.set(0)
            logger.debug("Cache cleared")
        }
    }
    
    override fun size(): Int = cache.size
    
    override fun stats(): CacheStats = CacheStats(
        hits = hits.get(),
        misses = misses.get(),
        size = cache.size,
        capacity = maxSize
    )
    
    private fun evictOldestEntry() {
        var toEvict: K? = null
        while (toEvict == null && accessQueue.isNotEmpty()) {
            // Poll the oldest entry from the queue
            val candidate = accessQueue.poll()
            // Only evict if it's still in the cache (could have been removed)
            if (cache.containsKey(candidate)) {
                toEvict = candidate
            }
        }
        
        toEvict?.let { 
            cache.remove(it)
            logger.debug("Evicted from cache: {}", it)
        }
    }
    
    private fun updateAccessOrder(key: K) {
        // Remove key if it exists and add it to the end of the queue
        accessQueue.remove(key)
        accessQueue.add(key)
    }
}

/**
 * Disk-based cache implementation
 *
 * @param directory Directory to store cache files
 * @param serializer Function to serialize values to ByteArray
 * @param deserializer Function to deserialize ByteArray to values
 * @param keyToFilename Function to convert keys to filenames
 * @param maxEntries Maximum number of entries to track (LRU eviction applies)
 */
class DiskCache<K, V>(
    private val directory: File,
    private val serializer: (V) -> ByteArray,
    private val deserializer: (ByteArray) -> V,
    private val keyToFilename: (K) -> String,
    private val maxEntries: Int = 1000
) : Cache<K, V> {
    private val logger = LoggerFactory.getLogger(DiskCache::class.java)
    
    private val accessQueue = ConcurrentLinkedQueue<K>()
    private val keyMap = ConcurrentHashMap<K, String>() // Maps keys to filenames
    private val hits = AtomicInteger(0)
    private val misses = AtomicInteger(0)
    private val mutex = Mutex()
    
    init {
        if (!directory.exists()) {
            directory.mkdirs()
        }
    }
    
    override suspend fun get(key: K): V? {
        val filename = keyMap[key] ?: keyToFilename(key)
        val file = File(directory, filename)
        
        if (!file.exists()) {
            misses.incrementAndGet()
            logger.debug("Cache miss for key: {}", key)
            return null
        }
        
        return try {
            val bytes = file.readBytes()
            val value = deserializer(bytes)
            updateAccessOrder(key)
            hits.incrementAndGet()
            logger.debug("Cache hit for key: {}", key)
            value
        } catch (e: Exception) {
            logger.error("Error reading from cache: {}", e.message)
            misses.incrementAndGet()
            null
        }
    }
    
    override suspend fun put(key: K, value: V) {
        mutex.withLock {
            // Ensure we're under capacity
            if (keyMap.size >= maxEntries && !keyMap.containsKey(key)) {
                evictOldestEntry()
            }
            
            val filename = keyToFilename(key)
            val file = File(directory, filename)
            
            try {
                val bytes = serializer(value)
                file.writeBytes(bytes)
                keyMap[key] = filename
                updateAccessOrder(key)
                logger.debug("Added to cache: {}", key)
            } catch (e: Exception) {
                logger.error("Error writing to cache: {}", e.message)
                file.delete()
            }
        }
    }
    
    override suspend fun remove(key: K): Boolean {
        mutex.withLock {
            val filename = keyMap.remove(key) ?: return false
            accessQueue.remove(key)
            val file = File(directory, filename)
            return file.delete()
        }
    }
    
    override suspend fun clear() {
        mutex.withLock {
            keyMap.keys.forEach { key ->
                val filename = keyMap[key] ?: return@forEach
                val file = File(directory, filename)
                file.delete()
            }
            
            keyMap.clear()
            accessQueue.clear()
            hits.set(0)
            misses.set(0)
            logger.debug("Cache cleared")
        }
    }
    
    override fun size(): Int = keyMap.size
    
    override fun stats(): CacheStats = CacheStats(
        hits = hits.get(),
        misses = misses.get(),
        size = keyMap.size,
        capacity = maxEntries
    )
    
    private fun evictOldestEntry() {
        var toEvict: K? = null
        while (toEvict == null && accessQueue.isNotEmpty()) {
            val candidate = accessQueue.poll()
            if (keyMap.containsKey(candidate)) {
                toEvict = candidate
            }
        }
        
        toEvict?.let {
            val filename = keyMap.remove(it) ?: return
            val file = File(directory, filename)
            file.delete()
            logger.debug("Evicted from cache: {}", it)
        }
    }
    
    private fun updateAccessOrder(key: K) {
        accessQueue.remove(key)
        accessQueue.add(key)
    }
}

/**
 * Multi-level cache that combines multiple caches with different characteristics
 * 
 * Typically used to combine a fast in-memory cache with a slower but larger disk cache
 */
class MultiLevelCache<K, V>(
    private val levels: List<Pair<Cache<K, V>, Int>>, // Cache instances with their priority (lower = higher priority)
    private val propagateWrites: Boolean = true // Whether writes to one level should propagate to other levels
) : Cache<K, V> {
    private val logger = LoggerFactory.getLogger(MultiLevelCache::class.java)
    
    // Sort levels by priority
    private val sortedLevels = levels.sortedBy { it.second }
    
    override suspend fun get(key: K): V? {
        // Try each cache level from highest to lowest priority
        for ((cache, _) in sortedLevels) {
            val value = cache.get(key)
            if (value != null) {
                // If found, propagate to higher-priority caches
                if (propagateWrites) {
                    propagateToHigherLevels(key, value, cache)
                }
                return value
            }
        }
        
        return null
    }
    
    override suspend fun put(key: K, value: V) {
        // If propagation is enabled, write to all caches
        if (propagateWrites) {
            for ((cache, _) in sortedLevels) {
                cache.put(key, value)
            }
        } else {
            // Otherwise, just write to the highest-priority cache
            sortedLevels.firstOrNull()?.first?.put(key, value)
        }
    }
    
    override suspend fun remove(key: K): Boolean {
        var removedAny = false
        
        // Remove from all caches
        for ((cache, _) in sortedLevels) {
            val removed = cache.remove(key)
            removedAny = removedAny || removed
        }
        
        return removedAny
    }
    
    override suspend fun clear() {
        // Clear all caches
        for ((cache, _) in sortedLevels) {
            cache.clear()
        }
    }
    
    override fun size(): Int {
        // Return the size of the largest cache
        return sortedLevels.maxOfOrNull { (cache, _) -> cache.size() } ?: 0
    }
    
    override fun stats(): CacheStats {
        // Combine stats from all caches
        var totalHits = 0
        var totalMisses = 0
        var totalSize = 0
        var totalCapacity = 0
        
        for ((cache, _) in sortedLevels) {
            val stats = cache.stats()
            totalHits += stats.hits
            totalMisses += stats.misses
            totalSize += stats.size
            totalCapacity += stats.capacity
        }
        
        return CacheStats(
            hits = totalHits,
            misses = totalMisses,
            size = totalSize,
            capacity = totalCapacity
        )
    }
    
    /**
     * Propagate a value to higher-priority caches
     */
    private suspend fun propagateToHigherLevels(key: K, value: V, foundInCache: Cache<K, V>) {
        val foundPriority = sortedLevels.find { it.first === foundInCache }?.second ?: return
        
        // Propagate to caches with higher priority (lower priority value)
        for ((cache, priority) in sortedLevels) {
            if (priority < foundPriority) {
                cache.put(key, value)
            }
        }
    }
    
    companion object {
        /**
         * Create a two-level memory + disk cache
         */
        fun <K : Serializable, V : Serializable> createTwoLevel(
            memorySize: Int = 1000,
            diskDirectory: File,
            keyToFilename: (K) -> String,
            diskSize: Int = 10000
        ): MultiLevelCache<K, V> {
            val memoryCache = MemoryCache<K, V>(memorySize)
            
            val diskCache = DiskCache(
                directory = diskDirectory,
                serializer = { value -> serializeObject(value) },
                deserializer = { bytes -> deserializeObject<V>(bytes) },
                keyToFilename = keyToFilename,
                maxEntries = diskSize
            )
            
            return MultiLevelCache(
                listOf(
                    memoryCache to 1, // Higher priority (lower number)
                    diskCache to 2     // Lower priority
                ),
                propagateWrites = true
            )
        }
        
        private fun <T : Serializable> serializeObject(obj: T): ByteArray {
            val baos = ByteArrayOutputStream()
            ObjectOutputStream(baos).use { it.writeObject(obj) }
            return baos.toByteArray()
        }
        
        private fun <T : Serializable> deserializeObject(bytes: ByteArray): T {
            val bais = ByteArrayInputStream(bytes)
            @Suppress("UNCHECKED_CAST")
            return ObjectInputStream(bais).use { it.readObject() as T }
        }
    }
}
