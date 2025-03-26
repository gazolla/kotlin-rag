package com.gazapps.rag.core

import com.gazapps.rag.core.reranking.RerankingOptions
import com.gazapps.rag.core.GenerationOptions

/**
 * Options for customizing RAG queries.
 *
 * @property filter Optional metadata filter to apply when retrieving documents
 * @property retrievalLimit Optional override for the number of documents to retrieve
 * @property includeMetadata Optional override for whether to include metadata in the context
 * @property rerank Whether to apply reranking to retrieved documents
 * @property rerankingOptions Options for the reranking process
 * @property generationOptions Options for the text generation
 */
data class QueryOptions(
    val filter: Map<String, Any>? = null,
    val retrievalLimit: Int? = null,
    val includeMetadata: Boolean? = null,
    val rerank: Boolean = false,
    val rerankingOptions: RerankingOptions? = null,
    val generationOptions: GenerationOptions? = null
) {
    /**
     * Create a copy of the options with a metadata filter
     */
    fun withFilter(filter: Map<String, Any>): QueryOptions {
        return copy(filter = filter)
    }
    
    /**
     * Create a copy of the options with a single metadata filter key-value pair
     */
    fun withFilter(key: String, value: Any): QueryOptions {
        return copy(filter = mapOf(key to value))
    }
    
    /**
     * Create a copy of the options with a retrieval limit
     */
    fun withRetrievalLimit(limit: Int): QueryOptions {
        return copy(retrievalLimit = limit)
    }
    
    /**
     * Create a copy of the options with reranking enabled
     */
    fun withReranking(
        enabled: Boolean = true,
        options: RerankingOptions? = null
    ): QueryOptions {
        return copy(rerank = enabled, rerankingOptions = options)
    }
    
    /**
     * Create a copy of the options with generation options
     */
    fun withGenerationOptions(options: GenerationOptions): QueryOptions {
        return copy(generationOptions = options)
    }
    
    companion object {
        /**
         * Create options with a metadata filter
         */
        fun withFilter(filter: Map<String, Any>): QueryOptions {
            return QueryOptions(filter = filter)
        }
        
        /**
         * Create options with a single metadata filter key-value pair
         */
        fun withFilter(key: String, value: Any): QueryOptions {
            return QueryOptions(filter = mapOf(key to value))
        }
    }
}
