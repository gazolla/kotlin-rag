package com.gazapps.rag.core.embedder

import com.gazapps.rag.core.DummyEmbedder
import kotlinx.coroutines.runBlocking
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull

class EmbeddingCacheTest {
    
    @Test
    fun `InMemoryEmbeddingCache should cache and retrieve embeddings`() = runBlocking {
        // Setup
        val cache = InMemoryEmbeddingCache(maxSize = 5)
        val text = "Test text for caching"
        val embedding = FloatArray(3) { it.toFloat() }
        
        // Act
        cache.put(text, embedding)
        val retrieved = cache.get(text)
        
        // Assert
        assertNotNull(retrieved)
        assertEquals(embedding.toList(), retrieved.toList())
    }
    
    @Test
    fun `InMemoryEmbeddingCache should respect max size and evict oldest entries`() = runBlocking {
        // Setup
        val cache = InMemoryEmbeddingCache(maxSize = 3)
        
        // Add 3 entries
        for (i in 1..3) {
            val text = "Text $i"
            val embedding = FloatArray(3) { i.toFloat() }
            cache.put(text, embedding)
        }
        
        // Verify all 3 are cached
        for (i in 1..3) {
            val retrieved = cache.get("Text $i")
            assertNotNull(retrieved)
        }
        
        // Add 2 more entries to trigger eviction
        for (i in 4..5) {
            val text = "Text $i"
            val embedding = FloatArray(3) { i.toFloat() }
            cache.put(text, embedding)
        }
        
        // Verify oldest entries were evicted
        assertNull(cache.get("Text 1"))
        assertNull(cache.get("Text 2"))
        
        // Verify newest entries are still cached
        for (i in 3..5) {
            val retrieved = cache.get("Text $i")
            assertNotNull(retrieved)
        }
    }
    
    @Test
    fun `CachedEmbedder should use cache for repeat queries`() = runBlocking {
        // Setup
        val cache = InMemoryEmbeddingCache()
        val mockEmbedder = DummyEmbedder(deterministic = true)
        val cachedEmbedder = CachedEmbedder(mockEmbedder, cache)
        
        // First call should not be in cache
        val text = "Test caching"
        val firstResult = cachedEmbedder.embed(text)
        
        // Verify cache stats - should be one miss, zero hits
        val statsAfterFirst = cachedEmbedder.getCacheStats()
        assertEquals(1, statsAfterFirst["size"])
        assertEquals(0.0, statsAfterFirst["hitRate"])
        
        // Second call should use cache
        val secondResult = cachedEmbedder.embed(text)
        
        // Verify results are the same
        assertEquals(firstResult.toList(), secondResult.toList())
        
        // Verify cache stats - should be one hit, one miss
        val statsAfterSecond = cachedEmbedder.getCacheStats()
        assertEquals(1, statsAfterSecond["size"])
        assertEquals(0.5, statsAfterSecond["hitRate"])
    }
    
    @Test
    fun `CachedEmbedder batchEmbed should use cache for known items`() = runBlocking {
        // Setup
        val cache = InMemoryEmbeddingCache()
        val mockEmbedder = DummyEmbedder(deterministic = true)
        val cachedEmbedder = CachedEmbedder(mockEmbedder, cache)
        
        // Pre-cache some embeddings
        val text1 = "Text one"
        val text2 = "Text two"
        cachedEmbedder.embed(text1)
        
        // Now batch embed both texts
        val texts = listOf(text1, text2)
        val embeddings = cachedEmbedder.batchEmbed(texts)
        
        // Verify we got 2 embeddings
        assertEquals(2, embeddings.size)
        
        // Verify cache stats - should be one hit, two misses
        val stats = cachedEmbedder.getCacheStats()
        assertEquals(2, stats["size"])
        assertEquals(1.0/3.0, stats["hitRate"])
    }
    
    @Test
    fun `CachedEmbedder should clear cache on demand`() = runBlocking {
        // Setup
        val cache = InMemoryEmbeddingCache()
        val mockEmbedder = DummyEmbedder()
        val cachedEmbedder = CachedEmbedder(mockEmbedder, cache)
        
        // Add some items to cache
        cachedEmbedder.embed("Item 1")
        cachedEmbedder.embed("Item 2")
        
        // Verify cache has items
        val statsBefore = cachedEmbedder.getCacheStats()
        assertEquals(2, statsBefore["size"])
        
        // Clear cache
        cachedEmbedder.clearCache()
        
        // Verify cache is empty
        val statsAfter = cachedEmbedder.getCacheStats()
        assertEquals(0, statsAfter["size"])
    }
}
