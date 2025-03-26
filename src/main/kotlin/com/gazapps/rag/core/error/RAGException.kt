package com.gazapps.rag.core.error

/**
 * Base exception class for RAG library
 */
sealed class RAGException(
    message: String,
    cause: Throwable? = null
) : Exception(message, cause) {

    /**
     * Exception during embedding process
     */
    class EmbeddingException(
        message: String, 
        cause: Throwable? = null,
        val documentId: String? = null
    ) : RAGException("Embedding error: $message", cause)

    /**
     * Exception during vector store operations
     */
    class VectorStoreException(
        message: String, 
        cause: Throwable? = null,
        val operation: String = ""
    ) : RAGException("Vector store error during $operation: $message", cause)

    /**
     * Exception during LLM invocation
     */
    class LLMException(
        message: String, 
        cause: Throwable? = null,
        val model: String? = null
    ) : RAGException("LLM error${model?.let { " with model $it" } ?: ""}: $message", cause)

    /**
     * Exception during document processing
     */
    class DocumentProcessingException(
        message: String,
        cause: Throwable? = null,
        val documentId: String? = null
    ) : RAGException("Document processing error${documentId?.let { " for document $it" } ?: ""}: $message", cause)

    /**
     * Exception for invalid configuration
     */
    class ConfigurationException(
        message: String,
        val property: String? = null
    ) : RAGException("Invalid configuration${property?.let { " for property $it" } ?: ""}: $message")

    /**
     * Exception during chunking operations
     */
    class ChunkingException(
        message: String,
        cause: Throwable? = null,
        val strategy: String? = null
    ) : RAGException("Chunking error${strategy?.let { " using strategy $it" } ?: ""}: $message", cause)

    /**
     * Time-out exception
     */
    class TimeoutException(
        message: String,
        val operationName: String = "",
        val timeoutMs: Long = 0
    ) : RAGException("Operation $operationName timed out after ${timeoutMs}ms: $message")

    /**
     * Rate-limit exception
     */
    class RateLimitException(
        message: String,
        val service: String = "",
        val retryAfterMs: Long? = null
    ) : RAGException("Rate limited by $service: $message${retryAfterMs?.let { ", retry after ${it}ms" } ?: ""}")

    /**
     * External service unavailable
     */
    class ServiceUnavailableException(
        message: String,
        val service: String = ""
    ) : RAGException("Service $service unavailable: $message")
}
