package com.gazapps.rag.core

import com.gazapps.rag.core.config.RAGConfig
import com.gazapps.rag.core.document.*
import com.gazapps.rag.core.error.*
import com.gazapps.rag.core.monitoring.RAGMetrics
import com.gazapps.rag.core.monitoring.RAGMetricsManager
import com.gazapps.rag.core.monitoring.time
import com.gazapps.rag.core.reranking.*
import kotlinx.coroutines.withTimeout
import java.io.File

/**
 * Enhanced RAG class with error handling, logging, metrics and recovery mechanisms
 */
class RAGWithErrorHandling(
    val embedder: Embedder,
    val vectorStore: VectorStore,
    val llmClient: LLMClient,
    val config: RAGConfig = RAGConfig(),
    private val metrics: RAGMetrics = RAGMetricsManager.getMetrics(),
    private val logger: RAGLogger = RAGLoggerFactory.getLogger(),
    private val errorHandler: ErrorHandler = ErrorHandler("RAGWithErrorHandling"),
    private val fallbackEmbedder: Embedder? = null,
    private val fallbackVectorStore: VectorStore? = null,
    private val fallbackLLMClient: LLMClient? = null,
    private val reranker: Reranker? = null,
    private val chunkingManager: ChunkingManager? = null
) : IRAG {
    // Circuit breakers for external services
    private val embeddingCircuitBreaker = CircuitBreakerRegistry.getCircuitBreaker("embedding")
    private val vectorStoreCircuitBreaker = CircuitBreakerRegistry.getCircuitBreaker("vectorstore")
    private val llmCircuitBreaker = CircuitBreakerRegistry.getCircuitBreaker("llm")
    
    /**
     * Index a document with error handling
     */
    override suspend fun indexDocument(document: Document): Boolean {
        logger.info(
            "Indexing document",
            "RAG",
            context = mapOf("documentId" to document.id, "contentLength" to document.content.length)
        )
        
        metrics.incrementCounter("rag.index.document.attempts")
        
        return try {
            metrics.time("rag.index.document") {
                // Step 1: Get embedding with circuit breaker and fallback
                val embedding = errorHandler.withCircuitBreaker(embeddingCircuitBreaker) {
                    errorHandler.withFallback(
                        primary = {
                            errorHandler.withRetry {
                                embedder.embed(document.content)
                            }
                        },
                        fallback = { exception ->
                            logger.warn(
                                "Using fallback embedder",
                                "RAG",
                                exception,
                                mapOf("documentId" to document.id)
                            )
                            fallbackEmbedder?.embed(document.content)
                                ?: throw RAGException.EmbeddingException(
                                    "Primary embedder failed and no fallback available",
                                    exception,
                                    document.id
                                )
                        }
                    )
                }
                
                // Step 2: Store with circuit breaker and fallback
                errorHandler.withCircuitBreaker(vectorStoreCircuitBreaker) {
                    errorHandler.withFallback(
                        primary = {
                            errorHandler.withRetry {
                                vectorStore.store(document, embedding)
                            }
                        },
                        fallback = { exception ->
                            logger.warn(
                                "Using fallback vector store",
                                "RAG",
                                exception,
                                mapOf("documentId" to document.id)
                            )
                            fallbackVectorStore?.store(document, embedding)
                                ?: throw RAGException.VectorStoreException(
                                    "Primary vector store failed and no fallback available",
                                    exception,
                                    "store"
                                )
                        }
                    )
                }
                
                metrics.incrementCounter("rag.index.document.success")
                logger.info(
                    "Document indexed successfully",
                    "RAG",
                    context = mapOf("documentId" to document.id)
                )
                true
            }
        } catch (e: Exception) {
            metrics.incrementCounter("rag.index.document.failures")
            logger.error(
                "Failed to index document",
                "RAG",
                e,
                mapOf("documentId" to document.id)
            )
            
            // Wrap in RAGException if needed
            if (e !is RAGException) {
                e.log(LogLevel.ERROR, "RAG", "Unexpected error during document indexing", 
                    mapOf("documentId" to document.id))
            }
            
            false
        }
    }
    
    /**
     * Index multiple documents with error handling
     */
    override suspend fun indexDocuments(documents: List<Document>): List<Boolean> {
        if (documents.isEmpty()) return emptyList()
        
        logger.info(
            "Indexing multiple documents",
            "RAG",
            context = mapOf("count" to documents.size)
        )
        
        metrics.incrementCounter("rag.index.batch.attempts")
        metrics.setGauge("rag.index.batch.size", documents.size.toLong())
        
        return try {
            metrics.time("rag.index.batch") {
                // Get embeddings with circuit breaker and fallback
                val embeddings = errorHandler.withCircuitBreaker(embeddingCircuitBreaker) {
                    errorHandler.withFallback(
                        primary = {
                            errorHandler.withRetry {
                                embedder.batchEmbed(documents.map { it.content })
                            }
                        },
                        fallback = { exception ->
                            logger.warn(
                                "Using fallback embedder for batch",
                                "RAG",
                                exception,
                                mapOf("documentCount" to documents.size)
                            )
                            fallbackEmbedder?.batchEmbed(documents.map { it.content })
                                ?: throw RAGException.EmbeddingException(
                                    "Primary embedder failed and no fallback available for batch",
                                    exception
                                )
                        }
                    )
                }
                
                // Store with circuit breaker and fallback
                errorHandler.withCircuitBreaker(vectorStoreCircuitBreaker) {
                    errorHandler.withFallback(
                        primary = {
                            errorHandler.withRetry {
                                vectorStore.batchStore(documents, embeddings)
                            }
                        },
                        fallback = { exception ->
                            logger.warn(
                                "Using fallback vector store for batch",
                                "RAG",
                                exception,
                                mapOf("documentCount" to documents.size)
                            )
                            fallbackVectorStore?.batchStore(documents, embeddings)
                                ?: throw RAGException.VectorStoreException(
                                    "Primary vector store failed and no fallback available for batch",
                                    exception,
                                    "batchStore"
                                )
                        }
                    )
                }
                
                metrics.incrementCounter("rag.index.batch.success")
                logger.info(
                    "Batch indexed successfully",
                    "RAG",
                    context = mapOf("documentCount" to documents.size)
                )
                List(documents.size) { true }
            }
        } catch (e: Exception) {
            metrics.incrementCounter("rag.index.batch.failures")
            logger.error(
                "Failed to index batch",
                "RAG",
                e,
                mapOf("documentCount" to documents.size)
            )
            
            // Wrap in RAGException if needed
            if (e !is RAGException) {
                e.log(LogLevel.ERROR, "RAG", "Unexpected error during batch indexing", 
                    mapOf("documentCount" to documents.size))
            }
            
            List(documents.size) { false }
        }
    }
    
    /**
     * Query the RAG system with error handling
     */
    override suspend fun query(question: String): RAGResponse {
        return query(question, null, null)
    }
    
    /**
     * Query with filter and error handling
     */
    override suspend fun query(question: String, filter: Map<String, Any>?): RAGResponse {
        return query(question, filter, null)
    }
    
    /**
     * Query with options and error handling
     */
    override suspend fun query(question: String, options: QueryOptions): RAGResponse {
        return query(question, options.filter, options)
    }
    
    /**
     * Internal query implementation with error handling
     */
    private suspend fun query(question: String, filter: Map<String, Any>?, options: QueryOptions?): RAGResponse {
        logger.info(
            "Processing query",
            "RAG",
            context = mapOf(
                "question" to question,
                "filter" to (filter?.toString() ?: "none"),
                "options" to (options?.toString() ?: "default")
            )
        )
        
        metrics.incrementCounter("rag.query.attempts")
        
        return try {
            metrics.time("rag.query.total") {
                // Step 1: Convert question to embedding
                val questionEmbedding = metrics.time("rag.query.embedding") {
                    errorHandler.withCircuitBreaker(embeddingCircuitBreaker) {
                        errorHandler.withFallback(
                            primary = {
                                errorHandler.withRetry {
                                    embedder.embed(question)
                                }
                            },
                            fallback = { exception ->
                                logger.warn(
                                    "Using fallback embedder for query",
                                    "RAG",
                                    exception
                                )
                                fallbackEmbedder?.embed(question)
                                    ?: throw RAGException.EmbeddingException(
                                        "Primary embedder failed and no fallback available for query",
                                        exception
                                    )
                            }
                        )
                    }
                }
                
                // Step 2: Retrieve documents
                val retrievalLimit = options?.retrievalLimit ?: config.retrieval.retrievalLimit
                
                val documents = metrics.time("rag.query.retrieval") {
                    errorHandler.withCircuitBreaker(vectorStoreCircuitBreaker) {
                        errorHandler.withFallback(
                            primary = {
                                errorHandler.withRetry {
                                    vectorStore.search(
                                        query = questionEmbedding,
                                        limit = retrievalLimit,
                                        filter = filter
                                    )
                                }
                            },
                            fallback = { exception ->
                                logger.warn(
                                    "Using fallback vector store for query",
                                    "RAG",
                                    exception
                                )
                                fallbackVectorStore?.search(
                                    query = questionEmbedding,
                                    limit = retrievalLimit,
                                    filter = filter
                                ) ?: throw RAGException.VectorStoreException(
                                    "Primary vector store failed and no fallback available for query",
                                    exception,
                                    "search"
                                )
                            }
                        )
                    }
                }
                
                // Record document retrieval metrics
                metrics.setGauge("rag.query.documents.retrieved", documents.size.toLong())
                
                // Step 3: Apply reranking if enabled
                val rerankedDocuments = if (config.retrieval.reranking || options?.rerank == true) {
                    metrics.time("rag.query.reranking") {
                        val reranker = getReranker()
                        reranker.rerank(
                            queryEmbedding = questionEmbedding,
                            documents = documents,
                            options = config.retrieval.rerankingOptions
                        )
                    }
                } else {
                    documents
                }
                
                // Step 4: Build context and prompt
                val context = buildContext(rerankedDocuments, question, options)
                val prompt = buildPrompt(question, context)
                
                // Step 5: Generate answer
                val answer = metrics.time("rag.query.generation") {
                    errorHandler.withCircuitBreaker(llmCircuitBreaker) {
                        errorHandler.withFallback(
                            primary = {
                                errorHandler.withRetry {
                                    llmClient.generate(prompt)
                                }
                            },
                            fallback = { exception ->
                                logger.warn(
                                    "Using fallback LLM for query",
                                    "RAG",
                                    exception
                                )
                                fallbackLLMClient?.generate(prompt)
                                    ?: throw RAGException.LLMException(
                                        "Primary LLM failed and no fallback available",
                                        exception
                                    )
                            }
                        )
                    }
                }
                
                metrics.incrementCounter("rag.query.success")
                logger.info(
                    "Query processed successfully",
                    "RAG",
                    context = mapOf(
                        "questionLength" to question.length,
                        "documentsRetrieved" to documents.size,
                        "answerLength" to answer.length
                    )
                )
                
                RAGResponse(
                    answer = answer,
                    documents = rerankedDocuments,
                    metadata = mapOf(
                        "questionEmbeddingDimensions" to questionEmbedding.size,
                        "documentsRetrieved" to documents.size,
                        "contextLength" to context.length,
                        "reranked" to (config.retrieval.reranking || options?.rerank == true)
                    )
                )
            }
        } catch (e: Exception) {
            metrics.incrementCounter("rag.query.failures")
            
            logger.error(
                "Query processing failed",
                "RAG",
                e,
                mapOf("question" to question)
            )
            
            // Create a fallback response
            val errorMessage = when (e) {
                is RAGException.EmbeddingException -> 
                    "Couldn't process your question. The embedding service is currently unavailable."
                is RAGException.VectorStoreException -> 
                    "Couldn't retrieve relevant information. The knowledge base is currently unavailable."
                is RAGException.LLMException -> 
                    "Found relevant information but couldn't generate a response. The AI service is currently unavailable."
                is RAGException.TimeoutException -> 
                    "The operation timed out. Please try again with a simpler question."
                else -> "An unexpected error occurred while processing your question. Please try again later."
            }
            
            RAGResponse(
                answer = errorMessage,
                documents = emptyList(),
                metadata = mapOf(
                    "error" to true,
                    "errorType" to e.javaClass.simpleName,
                    "errorMessage" to (e.message ?: "Unknown error")
                )
            )
        }
    }
    
    /**
     * Builds the context from retrieved documents
     */
    private fun buildContext(documents: List<ScoredDocument>, question: String, options: QueryOptions?): String {
        val includeMetadata = options?.includeMetadata ?: config.generation.includeMetadata
        
        return documents.joinToString("\n\n") { scoredDoc ->
            val doc = scoredDoc.document
            val content = doc.content
            
            if (includeMetadata && doc.metadata.isNotEmpty()) {
                val metadataStr = formatMetadata(doc.metadata)
                "$metadataStr\n\n$content"
            } else {
                content
            }
        }
    }
    
    /**
     * Format document metadata using the template
     */
    private fun formatMetadata(metadata: Map<String, Any>): String {
        var result = config.generation.metadataTemplate
        
        metadata.forEach { (key, value) ->
            result = result.replace("{$key}", value.toString())
        }
        
        // Replace any remaining template placeholders
        val placeholderRegex = Regex("\\{[^\\}]+\\}")
        result = result.replace(placeholderRegex, "")
        
        return result.trim()
    }
    
    /**
     * Builds the prompt combining context and question
     */
    private fun buildPrompt(question: String, context: String): String {
        return config.generation.promptTemplate
            .replace("{context}", context)
            .replace("{question}", question)
    }
    
    /**
     * Get or create a reranker based on configuration
     */
    private fun getReranker(): Reranker {
        return reranker ?: when (config.retrieval.rerankingStrategy) {
            RerankingStrategy.MMR -> MaximalMarginalRelevanceReranker()
            RerankingStrategy.CROSS_ENCODER -> CrossEncoderReranker()
            RerankingStrategy.ENSEMBLE -> {
                val mmrReranker = MaximalMarginalRelevanceReranker()
                EnsembleReranker(listOf(mmrReranker to 1.0f))
            }
            RerankingStrategy.HIERARCHICAL -> {
                // Hierarchical reranking (default to MMR for now)
                MaximalMarginalRelevanceReranker()
            }
            RerankingStrategy.NONE -> {
                object : Reranker {
                    override fun rerank(
                        queryEmbedding: FloatArray,
                        documents: List<ScoredDocument>,
                        options: RerankingOptions
                    ): List<ScoredDocument> = documents.take(options.limit)
                }
            }
        }
    }
    
    /**
     * Get or create a chunking manager
     */
    private fun getChunkingManager(): ChunkingManager {
        return chunkingManager ?: ChunkingManager(embedder)
    }
    
    /**
     * Index text content with error handling
     */
    override suspend fun indexFromText(content: String, id: String?, metadata: Map<String, Any>): Boolean {
        return indexFromText(content, id, metadata, true)
    }
    
    suspend fun indexFromText(
        content: String, 
        id: String?, 
        metadata: Map<String, Any>, 
        chunkContent: Boolean
    ): Boolean {
        logger.info(
            "Indexing from text",
            "RAG",
            context = mapOf(
                "contentLength" to content.length,
                "id" to (id ?: "auto-generated"),
                "chunkContent" to chunkContent
            )
        )
        
        metrics.incrementCounter("rag.index.text.attempts")
        
        try {
            // Preprocess the text if enabled
            val processedContent = if (config.indexing.preprocessText) {
                metrics.time("rag.index.text.preprocess") {
                    TextPreprocessor(config.indexing.textPreprocessingConfig).preprocess(content)
                }
            } else {
                content
            }
            
            val docId = id ?: generateDocumentId(processedContent)
            val document = SimpleDocument(docId, processedContent, metadata)
            
            val result = if (chunkContent) {
                indexDocumentWithChunking(document)
            } else {
                indexDocument(document)
            }
            
            if (result) {
                metrics.incrementCounter("rag.index.text.success")
            } else {
                metrics.incrementCounter("rag.index.text.failures")
            }
            
            return result
        } catch (e: Exception) {
            metrics.incrementCounter("rag.index.text.failures")
            
            logger.error(
                "Failed to index text",
                "RAG",
                e,
                mapOf(
                    "contentLength" to content.length,
                    "id" to (id ?: "auto-generated")
                )
            )
            
            // Wrap in RAGException if needed
            if (e !is RAGException) {
                throw RAGException.DocumentProcessingException(
                    "Failed to index text content",
                    e,
                    id
                )
            } else {
                throw e
            }
        }
    }
    
    /**
     * Index a document with chunking and error handling
     */
    override suspend fun indexDocumentWithChunking(document: Document): Boolean {
        logger.info(
            "Chunking and indexing document",
            "RAG",
            context = mapOf(
                "documentId" to document.id,
                "contentLength" to document.content.length,
                "chunkSize" to config.indexing.chunkSize,
                "strategy" to config.indexing.chunkingStrategy.name
            )
        )
        
        metrics.incrementCounter("rag.index.chunking.attempts")
        
        try {
            val chunks = metrics.time("rag.index.chunking") {
                try {
                    val chunker = getChunkingManager()
                    
                    // Create chunking configuration from RAG config
                    val chunkingConfig = ChunkingConfig(
                        chunkSize = config.indexing.chunkSize,
                        chunkOverlap = config.indexing.chunkOverlap,
                        strategy = config.indexing.chunkingStrategy,
                        preserveMetadata = true,
                        includeChunkMetadata = true
                    )
                    
                    // Chunk the document
                    chunker.chunkDocument(document, config.indexing.chunkingStrategy, chunkingConfig)
                } catch (e: Exception) {
                    logger.error(
                        "Failed to chunk document",
                        "RAG",
                        e,
                        mapOf("documentId" to document.id)
                    )
                    
                    throw RAGException.ChunkingException(
                        "Failed to chunk document",
                        e,
                        config.indexing.chunkingStrategy.name
                    )
                }
            }
            
            metrics.setGauge("rag.index.chunking.count", chunks.size.toLong())
            
            // Index all chunks
            val result = indexDocuments(chunks).all { it }
            
            if (result) {
                metrics.incrementCounter("rag.index.chunking.success")
                logger.info(
                    "Document chunked and indexed successfully",
                    "RAG",
                    context = mapOf(
                        "documentId" to document.id,
                        "chunkCount" to chunks.size
                    )
                )
            } else {
                metrics.incrementCounter("rag.index.chunking.failures")
                logger.warn(
                    "Some chunks failed to index",
                    "RAG",
                    context = mapOf(
                        "documentId" to document.id,
                        "chunkCount" to chunks.size
                    )
                )
            }
            
            return result
        } catch (e: Exception) {
            metrics.incrementCounter("rag.index.chunking.failures")
            
            logger.error(
                "Failed to chunk and index document",
                "RAG",
                e,
                mapOf("documentId" to document.id)
            )
            
            return false
        }
    }
    
    /**
     * Generate a document ID based on content
     */
    private fun generateDocumentId(content: String): String {
        val hash = content.hashCode().toString(16)
        return "doc-$hash"
    }
    
    /**
     * Index content from a file with error handling
     */
    override suspend fun indexFromFile(filePath: String, metadata: Map<String, Any>): Boolean {
        return indexFromFile(filePath, metadata, true)
    }
    
    suspend fun indexFromFile(
        filePath: String, 
        metadata: Map<String, Any>, 
        chunkContent: Boolean
    ): Boolean {
        logger.info(
            "Indexing from file",
            "RAG",
            context = mapOf(
                "filePath" to filePath,
                "chunkContent" to chunkContent
            )
        )
        
        metrics.incrementCounter("rag.index.file.attempts")
        
        try {
            val file = File(filePath)
            if (!file.exists() || !file.isFile) {
                logger.error(
                    "File does not exist or is not a file",
                    "RAG",
                    context = mapOf("filePath" to filePath)
                )
                
                metrics.incrementCounter("rag.index.file.notfound")
                return false
            }
            
            // Extract document metadata
            val fileMetadata = mapOf(
                "filename" to file.name,
                "filesize" to file.length(),
                "lastModified" to file.lastModified()
            ) + metadata
            
            // Use the appropriate extractor based on file extension
            val document = metrics.time("rag.index.file.extract") {
                try {
                    val extractor = DocumentExtractorFactory.getExtractorForFile(file.name)
                    extractor.extract(file.inputStream(), fileMetadata)
                } catch (e: Exception) {
                    logger.error(
                        "Failed to extract content from file",
                        "RAG",
                        e,
                        mapOf("filePath" to filePath)
                    )
                    
                    throw RAGException.DocumentProcessingException(
                        "Failed to extract content from file: ${e.message}",
                        e,
                        file.name
                    )
                }
            }
            
            val result = if (chunkContent) {
                indexDocumentWithChunking(document)
            } else {
                indexDocument(document)
            }
            
            if (result) {
                metrics.incrementCounter("rag.index.file.success")
                logger.info(
                    "File indexed successfully",
                    "RAG",
                    context = mapOf(
                        "filePath" to filePath,
                        "documentId" to document.id,
                        "contentLength" to document.content.length
                    )
                )
            } else {
                metrics.incrementCounter("rag.index.file.failures")
                logger.warn(
                    "Failed to index file",
                    "RAG",
                    context = mapOf("filePath" to filePath)
                )
            }
            
            return result
        } catch (e: Exception) {
            metrics.incrementCounter("rag.index.file.failures")
            
            logger.error(
                "Failed to index file",
                "RAG",
                e,
                mapOf("filePath" to filePath)
            )
            
            return false
        }
    }
}

/**
 * Builder for RAGWithErrorHandling instances
 */
class RAGWithErrorHandlingBuilder {
    var embedder: Embedder? = null
    var vectorStore: VectorStore? = null
    var llmClient: LLMClient? = null
    var fallbackEmbedder: Embedder? = null
    var fallbackVectorStore: VectorStore? = null
    var fallbackLLMClient: LLMClient? = null
    var config = RAGConfig()
    var metrics: RAGMetrics = RAGMetricsManager.getMetrics()
    var logger: RAGLogger = RAGLoggerFactory.getLogger()
    var errorHandler: ErrorHandler = ErrorHandler("RAGWithErrorHandling")
    var reranker: Reranker? = null
    var chunkingManager: ChunkingManager? = null
    
    /**
     * Configure the RAG instance
     */
    fun config(init: RAGConfig.() -> Unit) {
        config = RAGConfig().apply(init)
    }
    
    /**
     * Configure the embedding model
     */
    fun withEmbedder(embedder: Embedder): RAGWithErrorHandlingBuilder {
        this.embedder = embedder
        return this
    }
    
    /**
     * Configure the vector store
     */
    fun withVectorStore(vectorStore: VectorStore): RAGWithErrorHandlingBuilder {
        this.vectorStore = vectorStore
        return this
    }
    
    /**
     * Configure the LLM client
     */
    fun withLLMClient(llmClient: LLMClient): RAGWithErrorHandlingBuilder {
        this.llmClient = llmClient
        return this
    }
    
    /**
     * Configure a fallback embedder
     */
    fun withFallbackEmbedder(embedder: Embedder): RAGWithErrorHandlingBuilder {
        this.fallbackEmbedder = embedder
        return this
    }
    
    /**
     * Configure a fallback vector store
     */
    fun withFallbackVectorStore(vectorStore: VectorStore): RAGWithErrorHandlingBuilder {
        this.fallbackVectorStore = vectorStore
        return this
    }
    
    /**
     * Configure a fallback LLM client
     */
    fun withFallbackLLMClient(llmClient: LLMClient): RAGWithErrorHandlingBuilder {
        this.fallbackLLMClient = llmClient
        return this
    }
    
    /**
     * Configure custom metrics
     */
    fun withMetrics(metrics: RAGMetrics): RAGWithErrorHandlingBuilder {
        this.metrics = metrics
        return this
    }
    
    /**
     * Configure custom logger
     */
    fun withLogger(logger: RAGLogger): RAGWithErrorHandlingBuilder {
        this.logger = logger
        return this
    }
    
    /**
     * Configure custom error handler
     */
    fun withErrorHandler(errorHandler: ErrorHandler): RAGWithErrorHandlingBuilder {
        this.errorHandler = errorHandler
        return this
    }
    
    /**
     * Configure the reranker
     */
    fun withReranker(reranker: Reranker): RAGWithErrorHandlingBuilder {
        this.reranker = reranker
        return this
    }
    
    /**
     * Configure the chunking manager
     */
    fun withChunkingManager(chunkingManager: ChunkingManager): RAGWithErrorHandlingBuilder {
        this.chunkingManager = chunkingManager
        return this
    }
    
    /**
     * Build the RAG instance
     */
    fun build(): RAGWithErrorHandling {
        val actualEmbedder = embedder 
            ?: throw RAGException.ConfigurationException("Embedder must be provided", "embedder")
        
        val actualVectorStore = vectorStore 
            ?: throw RAGException.ConfigurationException("VectorStore must be provided", "vectorStore")
        
        val actualLLMClient = llmClient 
            ?: throw RAGException.ConfigurationException("LLMClient must be provided", "llmClient")
        
        return RAGWithErrorHandling(
            embedder = actualEmbedder,
            vectorStore = actualVectorStore,
            llmClient = actualLLMClient,
            fallbackEmbedder = fallbackEmbedder,
            fallbackVectorStore = fallbackVectorStore,
            fallbackLLMClient = fallbackLLMClient,
            config = config,
            metrics = metrics,
            logger = logger,
            errorHandler = errorHandler,
            reranker = reranker,
            chunkingManager = chunkingManager
        )
    }
}

/**
 * DSL function to create RAGWithErrorHandling instances
 */
fun ragWithErrorHandling(init: RAGWithErrorHandlingBuilder.() -> Unit): RAGWithErrorHandling {
    return RAGWithErrorHandlingBuilder().apply(init).build()
}
