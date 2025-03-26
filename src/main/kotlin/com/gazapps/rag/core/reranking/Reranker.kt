package com.gazapps.rag.core.reranking

import com.gazapps.rag.core.ScoredDocument
import com.gazapps.rag.core.VectorUtils

/**
 * Configuração para o processo de reranking.
 */
data class RerankingOptions(
    /**
     * Fator de diversidade para MMR (Maximum Marginal Relevance).
     * 0.0 = apenas diversidade, 1.0 = apenas relevância.
     */
    val diversityFactor: Float = 0.3f,
    
    /**
     * Lambda controla o balanço entre relevância e diversidade.
     * Alias para (1.0 - diversityFactor) para compatibilidade com implementações anteriores.
     */
    val lambda: Float = 0.7f,
    
    /**
     * Número máximo de documentos a retornar após reranking.
     */
    val limit: Int = 5,
    
    /**
     * Limiar mínimo de pontuação para documentos reranqueados.
     */
    val scoreThreshold: Float = 0.0f
)

/**
 * Interface para algoritmos de reranking.
 */
interface Reranker {
    /**
     * Reordena uma lista de documentos com base na similaridade com a consulta.
     *
     * @param queryEmbedding Embedding da consulta.
     * @param documents Lista de documentos a serem reordenados.
     * @param options Opções de configuração para o reranking.
     * @return Lista reordenada de documentos com pontuações atualizadas.
     */
    fun rerank(
        queryEmbedding: FloatArray,
        documents: List<ScoredDocument>,
        options: RerankingOptions = RerankingOptions()
    ): List<ScoredDocument>
    
    /**
     * Método de compatibilidade com a interface antiga
     */
    fun rerank(
        queryEmbedding: FloatArray,
        documents: List<ScoredDocument>,
        count: Int = documents.size
    ): List<ScoredDocument> {
        return rerank(queryEmbedding, documents, RerankingOptions(limit = count))
    }
}

/**
 * Implementa o algoritmo de Maximum Marginal Relevance (MMR) para reranking.
 * MMR equilibra relevância e diversidade na seleção de documentos.
 */
class MaximalMarginalRelevanceReranker(
    private val similarityFunction: (FloatArray, FloatArray) -> Float = VectorUtils::cosineSimilarity
) : Reranker {
    override fun rerank(
        queryEmbedding: FloatArray,
        documents: List<ScoredDocument>,
        options: RerankingOptions
    ): List<ScoredDocument>  {
        if (documents.isEmpty() || options.limit <= 0) {
            return emptyList()
        }
        
        if (documents.size <= 1) {
            return documents
        }
        
        // Lambda controla o balanço entre relevância e diversidade
        val lambda = options.lambda
        
        // Get document embeddings
        val documentEmbeddings = documents.mapNotNull { doc ->
            doc.document.metadata["embedding"] as? FloatArray
        }
        
        // If no embeddings are available in the metadata, fall back to simpler approach
        if (documentEmbeddings.size != documents.size) {
            return legacyRerank(queryEmbedding, documents, options)
        }
        
        val selected = mutableListOf<ScoredDocument>()
        val remaining = documents.toMutableList()
        
        // Adicionar o documento mais relevante primeiro
        val firstDoc = remaining.maxByOrNull { it.score }!!
        selected.add(firstDoc)
        remaining.remove(firstDoc)
        
        // Selecionar documentos iterativamente usando MMR
        while (selected.size < options.limit && remaining.isNotEmpty()) {
            var bestScore = Float.NEGATIVE_INFINITY
            var bestDoc: ScoredDocument? = null
            var bestIndex = -1
            
            for (i in remaining.indices) {
                val doc = remaining[i]
                val docEmbedding = documentEmbeddings[documents.indexOf(doc)]
                
                // Relevance score (similarity to query)
                val relevance = doc.score
                
                // Diversity score (how different this document is from already selected ones)
                var maxSimilarityToSelected = 0f
                for (selectedDoc in selected) {
                    val selectedIndex = documents.indexOf(selectedDoc)
                    val selectedEmbedding = documentEmbeddings[selectedIndex]
                    val similarity = similarityFunction(docEmbedding, selectedEmbedding)
                    maxSimilarityToSelected = maxOf(maxSimilarityToSelected, similarity)
                }
                
                // MMR score: λ * relevance - (1 - λ) * maxSimilarity
                val mmrScore = lambda * relevance - (1 - lambda) * maxSimilarityToSelected
                
                if (mmrScore > bestScore) {
                    bestScore = mmrScore
                    bestDoc = doc
                    bestIndex = i
                }
            }
            
            bestDoc?.let {
                // Atualizar o score do documento para refletir o score MMR
                val updatedDoc = ScoredDocument(it.document, bestScore)
                selected.add(updatedDoc)
                remaining.removeAt(bestIndex)
            } ?: break
        }
        
        // Filtrar documentos com score abaixo do limiar
        return selected
            .filter { it.score >= options.scoreThreshold }
            .sortedByDescending { it.score }
    }
    
    /**
     * Legacy implementation for when embeddings are not available directly.
     */
    private fun legacyRerank(
        queryEmbedding: FloatArray,
        documents: List<ScoredDocument>,
        options: RerankingOptions
    ): List<ScoredDocument> {
        val lambda = options.lambda
        
        val selected = mutableListOf<ScoredDocument>()
        val remaining = documents.toMutableList()
        
        // Adicionar o documento mais relevante primeiro
        val firstDoc = remaining.maxByOrNull { it.score }!!
        selected.add(firstDoc)
        remaining.remove(firstDoc)
        
        // Selecionar documentos iterativamente usando MMR
        while (selected.size < options.limit && remaining.isNotEmpty()) {
            var bestScore = Float.NEGATIVE_INFINITY
            var bestDoc: ScoredDocument? = null
            
            for (doc in remaining) {
                // Relevância - similaridade com a consulta (já calculada)
                val relevance = doc.score
                
                // Diversidade - 1 - máxima similaridade com documentos já selecionados
                var maxSimilarity = Float.NEGATIVE_INFINITY
                for (selectedDoc in selected) {
                    // Aqui seria ideal calcular a similaridade real entre os embeddings,
                    // mas por simplicidade vamos usar uma estimativa baseada nos scores
                    val similarity = doc.score * selectedDoc.score
                    if (similarity > maxSimilarity) {
                        maxSimilarity = similarity
                    }
                }
                
                // Se não encontramos similaridade, definir para 0
                if (maxSimilarity == Float.NEGATIVE_INFINITY) {
                    maxSimilarity = 0f
                }
                
                val diversity = 1.0f - maxSimilarity
                
                // Calcular score MMR
                val mmrScore = lambda * relevance + (1.0f - lambda) * diversity
                
                if (mmrScore > bestScore) {
                    bestScore = mmrScore
                    bestDoc = doc
                }
            }
            
            bestDoc?.let {
                // Atualizar o score do documento para refletir o score MMR
                val updatedDoc = ScoredDocument(it.document, bestScore)
                selected.add(updatedDoc)
                remaining.remove(it)
            } ?: break
        }
        
        // Filtrar documentos com score abaixo do limiar
        return selected
            .filter { it.score >= options.scoreThreshold }
            .sortedByDescending { it.score }
    }
}

/**
 * Alias para compatibilidade com códigos anteriores
 */
typealias MMRReranker = MaximalMarginalRelevanceReranker

/**
 * Implementação de reranking usando um modelo de cross-encoder.
 */
class CrossEncoderReranker : Reranker {
    override fun rerank(
        queryEmbedding: FloatArray,
        documents: List<ScoredDocument>,
        options: RerankingOptions
    ): List<ScoredDocument> {
        // Esta é uma implementação simplificada - numa implementação real,
        // seria usado um modelo cross-encoder para calcular a relevância
        
        // Por enquanto, apenas mantemos a ordem original
        return documents
            .take(options.limit)
            .filter { it.score >= options.scoreThreshold }
    }
}

/**
 * Reranker que combina múltiplos algoritmos de reranking.
 */
class EnsembleReranker(
    private val rerankers: List<Pair<Reranker, Float>>  // Reranker com seu peso
) : Reranker {
    override fun rerank(
        queryEmbedding: FloatArray,
        documents: List<ScoredDocument>,
        options: RerankingOptions
    ): List<ScoredDocument> {
        if (rerankers.isEmpty()) return documents
        
        // Aplicar cada reranker e combinar os resultados
        val scores = mutableMapOf<String, Float>()
        
        // Inicializar com zeros
        documents.forEach { doc ->
            scores[doc.document.id] = 0f
        }
        
        // Somar scores ponderados de cada reranker
        var totalWeight = 0f
        for ((reranker, weight) in rerankers) {
            val rerankedDocs = reranker.rerank(queryEmbedding, documents, options)
            
            for (doc in rerankedDocs) {
                val currentScore = scores[doc.document.id] ?: 0f
                scores[doc.document.id] = currentScore + (doc.score * weight)
            }
            
            totalWeight += weight
        }
        
        // Normalizar scores
        if (totalWeight > 0) {
            for (id in scores.keys) {
                scores[id] = scores[id]!! / totalWeight
            }
        }
        
        // Criar nova lista com scores atualizados
        val result = documents.map { doc ->
            val newScore = scores[doc.document.id] ?: doc.score
            ScoredDocument(doc.document, newScore)
        }
        
        // Filtrar e ordenar
        return result
            .filter { it.score >= options.scoreThreshold }
            .sortedByDescending { it.score }
            .take(options.limit)
    }
    
    companion object {
        fun of(vararg rerankers: Pair<Reranker, Float>): EnsembleReranker {
            return EnsembleReranker(rerankers.toList())
        }
    }
}

/**
 * Simple reranker that applies a boost to documents with specific metadata attributes
 */
class MetadataBoostReranker(
    private val boostFields: Map<String, Float>
) : Reranker {
    
    override fun rerank(
        queryEmbedding: FloatArray,
        documents: List<ScoredDocument>,
        options: RerankingOptions
    ): List<ScoredDocument> {
        if (documents.isEmpty()) {
            return emptyList()
        }
        
        // Apply boosts based on metadata
        val boostedDocuments = documents.map { scoredDoc ->
            var boostedScore = scoredDoc.score
            
            // Apply boosts for each field
            for ((field, boost) in boostFields) {
                if (scoredDoc.document.metadata.containsKey(field)) {
                    boostedScore *= boost
                }
            }
            
            ScoredDocument(scoredDoc.document, boostedScore)
        }
        
        // Sort by boosted score and return the top 'limit'
        return boostedDocuments
            .filter { it.score >= options.scoreThreshold }
            .sortedByDescending { it.score }
            .take(options.limit)
    }
}

/**
 * Reranker that combines multiple rerankers sequentially
 */
class CompositeReranker(
    private val rerankers: List<Reranker>
) : Reranker {
    
    override fun rerank(
        queryEmbedding: FloatArray,
        documents: List<ScoredDocument>,
        options: RerankingOptions
    ): List<ScoredDocument> {
        var currentDocuments = documents
        
        for (reranker in rerankers) {
            currentDocuments = reranker.rerank(queryEmbedding, currentDocuments, options)
        }
        
        return currentDocuments
    }
    
    companion object {
        fun of(vararg rerankers: Reranker): CompositeReranker {
            return CompositeReranker(rerankers.toList())
        }
    }
}
