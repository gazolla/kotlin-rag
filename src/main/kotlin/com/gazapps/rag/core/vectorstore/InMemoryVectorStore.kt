package com.gazapps.rag.core.vectorstore

import com.gazapps.rag.core.*
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicInteger

/**
 * Implementação simples de VectorStore que mantém todos os documentos e embeddings em memória.
 * Útil para testes e pequenas aplicações.
 */
class InMemoryVectorStore : VectorStore {
    // Armazenamento principal: ID do documento -> (Documento, Embedding)
    private val storage = ConcurrentHashMap<String, Pair<Document, FloatArray>>()
    
    // Contador para IDs gerados automaticamente
    private val idCounter = AtomicInteger(0)
    
    /**
     * Armazena um documento e seu embedding.
     */
    override suspend fun store(document: Document, embedding: FloatArray) {
        storage[document.id] = document to embedding
    }
    
    /**
     * Armazena múltiplos documentos e seus embeddings.
     */
    override suspend fun batchStore(documents: List<Document>, embeddings: List<FloatArray>) {
        require(documents.size == embeddings.size) { 
            "Number of documents (${documents.size}) must match number of embeddings (${embeddings.size})" 
        }
        
        documents.zip(embeddings).forEach { (doc, emb) ->
            storage[doc.id] = doc to emb
        }
    }
    
    /**
     * Busca documentos por similaridade.
     */
    override suspend fun search(
        query: FloatArray, 
        limit: Int, 
        filter: Map<String, Any>?
    ): List<ScoredDocument> {
        val candidates = if (filter != null && filter.isNotEmpty()) {
            // Aplicar filtro
            storage.values.filter { (doc, _) ->
                matchesFilter(doc, filter)
            }
        } else {
            // Sem filtro, usar todos os documentos
            storage.values.toList()
        }
        
        // Calcular similaridade para cada documento
        return candidates.map { (doc, emb) ->
            val similarity = VectorUtils.cosineSimilarity(query, emb)
            ScoredDocument(doc, similarity)
        }
        .filter { it.score > 0f } // Remover documentos sem similaridade
        .sortedByDescending { it.score } // Ordenar por similaridade
        .take(limit) // Limitar número de resultados
    }
    
    /**
     * Remove um documento.
     */
    override suspend fun delete(documentId: String) {
        storage.remove(documentId)
    }
    
    /**
     * Remove todos os documentos.
     */
    override suspend fun clear() {
        storage.clear()
    }
    
    /**
     * Retorna o número de documentos armazenados.
     */
    fun size(): Int = storage.size
    
    /**
     * Retorna uma lista de todos os documentos armazenados.
     * Útil para propósitos de demonstração e testes.
     */
    fun getStoredDocuments(): List<Document> {
        return storage.values.map { (document, _) -> document }
    }
    
    /**
     * Verifica se um documento corresponde ao filtro fornecido.
     */
    private fun matchesFilter(document: Document, filter: Map<String, Any>): Boolean {
        return filter.all { (key, value) ->
            document.metadata[key]?.let { docValue ->
                when (value) {
                    is String -> docValue.toString() == value
                    is Number -> (docValue as? Number)?.toDouble() == value.toDouble()
                    is Boolean -> docValue == value
                    is List<*> -> value.contains(docValue)
                    else -> docValue == value
                }
            } ?: false
        }
    }
}
