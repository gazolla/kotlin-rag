package com.gazapps.rag.core.config

import com.gazapps.rag.core.reranking.RerankingOptions
import com.gazapps.rag.core.RerankingStrategy

/**
 * Configuração específica para o processo de recuperação de documentos.
 */
data class RetrievalConfig(
    /**
     * Número de documentos a recuperar durante consultas.
     */
    var retrievalLimit: Int = 5,
    
    /**
     * Limiar mínimo de similaridade para documentos recuperados.
     */
    var similarityThreshold: Float = 0.7f,
    
    /**
     * Se deve aplicar reranking aos documentos recuperados.
     */
    var reranking: Boolean = false,
    
    /**
     * Opções de configuração para reranking.
     */
    var rerankingOptions: RerankingOptions = RerankingOptions(),
    
    /**
     * Estratégia a ser usada para reranking de documentos.
     */
    var rerankingStrategy: RerankingStrategy = RerankingStrategy.MMR
)
