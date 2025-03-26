package com.gazapps.rag.core.embedder

import com.gazapps.rag.core.Embedder
import java.io.File

/**
 * Esta implementação foi movida para MultiLevelEmbeddingCache.kt
 * @deprecated Use a implementação em MultiLevelEmbeddingCache.kt
 */
class SimpleCachedEmbedder(
    private val delegate: Embedder,
    private val memorySize: Int = 100,
    private val diskCacheDirectory: File? = null
) : Embedder {
    
    // Estatísticas de uso do cache
    private var hits = 0
    private var misses = 0
    
    // Cache em memória (primeiro nível)
    private val memoryCache = LinkedHashMap<String, FloatArray>(memorySize, 0.75f, true)
    
    /**
     * Gera embedding para um texto, usando cache quando disponível.
     */
    override suspend fun embed(text: String): FloatArray {
        val cacheKey = createCacheKey(text)
        
        // Verificar cache em memória primeiro
        val memoryCached = getCachedFromMemory(cacheKey)
        if (memoryCached != null) {
            hits++
            return memoryCached
        }
        
        // Verificar cache em disco se configurado
        val diskCached = diskCacheDirectory?.let { getCachedFromDisk(cacheKey, it) }
        if (diskCached != null) {
            hits++
            // Adicionar ao cache em memória também
            addToMemoryCache(cacheKey, diskCached)
            return diskCached
        }
        
        // Sem cache, gerar embedding
        misses++
        val embedding = delegate.embed(text)
        
        // Armazenar em ambos os caches
        addToMemoryCache(cacheKey, embedding)
        diskCacheDirectory?.let { addToDiskCache(cacheKey, embedding, it) }
        
        return embedding
    }
    
    /**
     * Gera embeddings para múltiplos textos, usando cache quando disponível.
     */
    override suspend fun batchEmbed(texts: List<String>): List<FloatArray> {
        // Implementação simples; uma versão otimizada processaria em lote os misses
        return texts.map { embed(it) }
    }
    
    /**
     * Retorna estatísticas de uso do cache.
     */
    fun getCacheStats(): CacheStats {
        val total = hits + misses
        val hitRate = if (total > 0) hits.toFloat() / total else 0f
        
        return CacheStats(hits, misses, hitRate)
    }
    
    /**
     * Cria uma chave de cache baseada no hash do texto.
     */
    private fun createCacheKey(text: String): String {
        return "emb_${text.hashCode().toString(16)}"
    }
    
    /**
     * Obtém um embedding do cache em memória.
     */
    private fun getCachedFromMemory(key: String): FloatArray? {
        synchronized(memoryCache) {
            return memoryCache[key]
        }
    }
    
    /**
     * Adiciona um embedding ao cache em memória, respeitando o limite de tamanho.
     */
    private fun addToMemoryCache(key: String, embedding: FloatArray) {
        synchronized(memoryCache) {
            memoryCache[key] = embedding
            
            // Manter o tamanho do cache dentro do limite
            while (memoryCache.size > memorySize) {
                val oldestKey = memoryCache.keys.firstOrNull() ?: break
                memoryCache.remove(oldestKey)
            }
        }
    }
    
    /**
     * Obtém um embedding do cache em disco.
     */
    private fun getCachedFromDisk(key: String, directory: File): FloatArray? {
        // Implementação simplificada; uma implementação real usaria serialização apropriada
        val cacheFile = File(directory, key)
        return if (cacheFile.exists()) {
            try {
                // Formato simples: valores de float separados por vírgula
                val data = cacheFile.readText()
                val values = data.split(",").map { it.toFloat() }.toFloatArray()
                values
            } catch (e: Exception) {
                null
            }
        } else {
            null
        }
    }
    
    /**
     * Adiciona um embedding ao cache em disco.
     */
    private fun addToDiskCache(key: String, embedding: FloatArray, directory: File) {
        try {
            directory.mkdirs()
            val cacheFile = File(directory, key)
            // Formato simples: valores de float separados por vírgula
            val data = embedding.joinToString(",")
            cacheFile.writeText(data)
        } catch (e: Exception) {
            // Ignorar erros de escrita em disco
        }
    }
    
    /**
     * Estatísticas de uso do cache.
     */
    data class CacheStats(
        val hits: Int,
        val misses: Int,
        val hitRate: Float
    )
}


