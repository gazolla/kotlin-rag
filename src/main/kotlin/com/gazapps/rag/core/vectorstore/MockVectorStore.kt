package com.gazapps.rag.core.vectorstore

import com.gazapps.rag.core.Document
import com.gazapps.rag.core.ScoredDocument
import com.gazapps.rag.core.VectorStore
import java.util.concurrent.ConcurrentHashMap
import kotlin.math.max

/**
 * Implementação simulada de VectorStore para testes.
 * Armazena documentos e embeddings em memória com comportamento determinístico.
 */
class MockVectorStore(
    private val similarityThreshold: Float = 0.7f,
    private val failureRate: Float = 0.0f // Taxa de falha simulada para testes (0.0 = sem falhas)
) : VectorStore {
    private val store = ConcurrentHashMap<String, Pair<Document, FloatArray>>()
    
    /**
     * Armazena um documento com seu embedding.
     */
    override suspend fun store(document: Document, embedding: FloatArray) {
        simulateFailure()
        store[document.id] = document to embedding
    }
    
    /**
     * Armazena múltiplos documentos com seus embeddings.
     */
    override suspend fun batchStore(documents: List<Document>, embeddings: List<FloatArray>) {
        simulateFailure()
        require(documents.size == embeddings.size) { 
            "Number of documents (${documents.size}) must match number of embeddings (${embeddings.size})" 
        }
        
        documents.zip(embeddings).forEach { (doc, emb) ->
            store[doc.id] = doc to emb
        }
    }
    
    /**
     * Busca por documentos similares a um embedding de consulta.
     */
    override suspend fun search(
        query: FloatArray, 
        limit: Int, 
        filter: Map<String, Any>?
    ): List<ScoredDocument> {
        simulateFailure()
        
        // Calcular similaridade com todos os documentos armazenados
        val results = store.values.map { (doc, emb) ->
            // Verificar filtros se fornecidos
            if (filter != null && !matchesFilter(doc, filter)) {
                return@map null
            }
            
            // Calcular similaridade de cosseno
            val similarity = cosineSimilarity(query, emb)
            
            // Apenas retornar documentos acima do limiar
            if (similarity >= similarityThreshold) {
                ScoredDocument(doc, similarity)
            } else {
                null
            }
        }
        .filterNotNull()
        .sortedByDescending { it.score }
        .take(limit)
        
        return results
    }
    
    /**
     * Remove um documento do armazenamento.
     */
    override suspend fun delete(documentId: String) {
        simulateFailure()
        store.remove(documentId)
    }
    
    /**
     * Limpa todos os documentos armazenados.
     */
    override suspend fun clear() {
        simulateFailure()
        store.clear()
    }
    
    /**
     * Calcula a similaridade de cosseno entre dois vetores.
     */
    private fun cosineSimilarity(a: FloatArray, b: FloatArray): Float {
        require(a.size == b.size) { "Vectors must have the same dimensions" }
        
        var dotProduct = 0f
        var normA = 0f
        var normB = 0f
        
        for (i in a.indices) {
            dotProduct += a[i] * b[i]
            normA += a[i] * a[i]
            normB += b[i] * b[i]
        }
        
        return if (normA > 0f && normB > 0f) {
            dotProduct / (kotlin.math.sqrt(normA) * kotlin.math.sqrt(normB))
        } else {
            0f
        }
    }
    
    /**
     * Verifica se um documento corresponde aos filtros de metadados.
     */
    private fun matchesFilter(document: Document, filter: Map<String, Any>): Boolean {
        return filter.all { (key, value) ->
            document.metadata[key] == value
        }
    }
    
    /**
     * Simula falhas ocasionais para testes de robustez.
     */
    private fun simulateFailure() {
        if (failureRate > 0 && Math.random() < failureRate) {
            throw RuntimeException("Simulated vector store failure")
        }
    }
    
    /**
     * Retorna o número de documentos armazenados.
     */
    fun count(): Int = store.size
}
