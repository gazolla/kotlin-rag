package com.gazapps.rag.core.config

import com.gazapps.rag.core.document.ChunkingStrategy
import com.gazapps.rag.core.document.PreprocessingConfig
import com.gazapps.rag.core.reranking.RerankingOptions
import com.gazapps.rag.core.RerankingStrategy

/**
 * Configuração completa para o sistema RAG
 */
data class RAGConfig(
    /**
     * Configuração para o processo de indexação.
     */
    var indexing: IndexingConfig = IndexingConfig(),
    
    /**
     * Configuração para o processo de recuperação.
     */
    var retrieval: RetrievalConfig = RetrievalConfig(),
    
    /**
     * Configuração para o processo de geração.
     */
    var generation: GenerationConfig = GenerationConfig(),
    
    /**
     * Se deve habilitar cache para otimização de desempenho.
     */
    var cacheEnabled: Boolean = true
) {
    /**
     * Para compatibilidade com código existente
     */
    // Propriedades delegadas para IndexingConfig
    var chunkSize: Int
        get() = indexing.chunkSize
        set(value) { indexing.chunkSize = value }
    
    var chunkOverlap: Int
        get() = indexing.chunkOverlap
        set(value) { indexing.chunkOverlap = value }
    
    var chunkingStrategy: ChunkingStrategy
        get() = indexing.chunkingStrategy
        set(value) { indexing.chunkingStrategy = value }
    
    var preprocessText: Boolean
        get() = indexing.preprocessText
        set(value) { indexing.preprocessText = value }
    
    var textPreprocessingConfig: PreprocessingConfig
        get() = indexing.textPreprocessingConfig
        set(value) { indexing.textPreprocessingConfig = value }
    
    var asyncProcessing: Boolean
        get() = indexing.asyncProcessing
        set(value) { indexing.asyncProcessing = value }
    
    var asyncBatchSize: Int
        get() = indexing.asyncBatchSize
        set(value) { indexing.asyncBatchSize = value }
    
    var asyncConcurrency: Int
        get() = indexing.asyncConcurrency
        set(value) { indexing.asyncConcurrency = value }
    
    var asyncTimeout: Long
        get() = indexing.asyncTimeout
        set(value) { indexing.asyncTimeout = value }
    
    // Propriedades delegadas para RetrievalConfig
    var retrievalLimit: Int
        get() = retrieval.retrievalLimit
        set(value) { retrieval.retrievalLimit = value }
    
    var similarityThreshold: Float
        get() = retrieval.similarityThreshold
        set(value) { retrieval.similarityThreshold = value }
    
    var reranking: Boolean
        get() = retrieval.reranking
        set(value) { retrieval.reranking = value }
    
    var rerankingOptions: RerankingOptions
        get() = retrieval.rerankingOptions
        set(value) { retrieval.rerankingOptions = value }
    
    var rerankingStrategy: RerankingStrategy
        get() = retrieval.rerankingStrategy
        set(value) { retrieval.rerankingStrategy = value }
    
    // Propriedades delegadas para GenerationConfig
    var promptTemplate: String
        get() = generation.promptTemplate
        set(value) { generation.promptTemplate = value }
    
    var includeMetadata: Boolean
        get() = generation.includeMetadata
        set(value) { generation.includeMetadata = value }
    
    var metadataTemplate: String
        get() = generation.metadataTemplate
        set(value) { generation.metadataTemplate = value }
    
    /**
     * Funções para configuração fluente
     */
    fun withIndexing(init: IndexingConfig.() -> Unit): RAGConfig {
        indexing.init()
        return this
    }
    
    fun withRetrieval(init: RetrievalConfig.() -> Unit): RAGConfig {
        retrieval.init()
        return this
    }
    
    fun withGeneration(init: GenerationConfig.() -> Unit): RAGConfig {
        generation.init()
        return this
    }
}
