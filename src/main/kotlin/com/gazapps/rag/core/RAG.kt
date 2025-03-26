package com.gazapps.rag.core

import com.gazapps.rag.core.config.RAGConfig
import com.gazapps.rag.core.document.*
import com.gazapps.rag.core.reranking.*
import com.gazapps.rag.core.utils.AsyncBatchProcessor
import kotlinx.coroutines.withTimeout
import kotlin.math.absoluteValue
import java.io.File
import kotlin.time.Duration

/**
 * Reranking strategies
 * @deprecated Use com.gazapps.rag.core.reranking.RerankingStrategy or com.gazapps.rag.core.config.RetrievalConfig
 */
@Deprecated("Use dedicated config classes")
enum class RerankingStrategy {
    NONE,   // No reranking
    MMR,    // Maximum Marginal Relevance
    CROSS_ENCODER, // Cross-encoder model
    ENSEMBLE, // Ensemble of multiple rerankers
    HIERARCHICAL // Hierarchical reranking
}

/**
 * Main RAG (Retrieval-Augmented Generation) class that coordinates
 * the process of retrieving relevant documents and generating responses
 */
class RAG(
    val embedder: Embedder,
    val vectorStore: VectorStore,
    val llmClient: LLMClient,
    val config: RAGConfig = RAGConfig(),
    private val reranker: Reranker? = null,
    private val chunkingManager: ChunkingManager? = null
) : IRAG {
    /**
     * Index a single document
     * 
     * @param document The document to index
     * @return Boolean indicating success
     */
    override suspend fun indexDocument(document: Document): Boolean {
        try {
            val embedding = embedder.embed(document.content)
            vectorStore.store(document, embedding)
            return true
        } catch (e: Exception) {
            // In a real implementation, we would log this error
            return false
        }
    }
    
    /**
     * Index multiple documents
     * 
     * @param documents List of documents to index
     * @return List of booleans indicating success for each document
     */
    override suspend fun indexDocuments(documents: List<Document>): List<Boolean> {
        if (documents.isEmpty()) return emptyList()
        
        if (config.indexing.asyncProcessing && documents.size > 1) {
            return indexDocumentsAsync(documents)
        }
        
        val embeddings = embedder.batchEmbed(documents.map { it.content })
        
        return try {
            vectorStore.batchStore(documents, embeddings)
            List(documents.size) { true }
        } catch (e: Exception) {
            // In a real implementation, we would log this error and handle partial failures
            List(documents.size) { false }
        }
    }
    
    /**
     * Index documents asynchronously in batches
     */
    private suspend fun indexDocumentsAsync(documents: List<Document>): List<Boolean> {
        val batchConfig = AsyncBatchProcessor.BatchConfig(
            concurrency = config.indexing.asyncConcurrency,
            timeout = Duration.parse("${config.indexing.asyncTimeout}ms"),
            maxRetries = 3
        )
        
        val results = AsyncBatchProcessor.processInBatches(
            items = documents,
            batchSize = config.indexing.asyncBatchSize,
            config = batchConfig,
            processor = { batch -> 
                val embeddings = embedder.batchEmbed(batch.map { it.content })
                vectorStore.batchStore(batch, embeddings)
                List(batch.size) { true }
            }
        )
        
        return results
    }
    
    /**
     * Query the RAG system with a question
     * 
     * @param question The question to ask
     * @return RAGResponse containing the answer and relevant documents
     */
    override suspend fun query(question: String): RAGResponse {
        return query(question, null, null)
    }
    
    /**
     * Query with additional filter
     * 
     * @param question The question to ask
     * @param filter Optional metadata filter
     * @return RAGResponse containing the answer and relevant documents
     */
    override suspend fun query(question: String, filter: Map<String, Any>?): RAGResponse {
        return query(question, filter, null)
    }
    
    /**
     * Query with options
     * 
     * @param question The question to ask
     * @param options Additional query options
     * @return RAGResponse containing the answer and relevant documents
     */
    override suspend fun query(question: String, options: QueryOptions): RAGResponse {
        return query(question, options.filter, options)
    }
    
    /**
     * Internal query implementation that handles all query variations
     */
    private suspend fun query(question: String, filter: Map<String, Any>?, options: QueryOptions?): RAGResponse {
        val startTime = System.currentTimeMillis()
        
        // Convert question to embedding
        val questionEmbedding = embedder.embed(question)
        
        // Determine retrieval limit
        val limit = options?.retrievalLimit ?: config.retrieval.retrievalLimit
        
        // Retrieve relevant documents
        val documents = vectorStore.search(
            query = questionEmbedding,
            limit = limit,
            filter = filter
        )
        
        // Apply reranking if enabled
        val rerankedDocuments = if (config.retrieval.reranking || options?.rerank == true) {
            val reranker = getReranker()
            reranker.rerank(
                queryEmbedding = questionEmbedding,
                documents = documents,
                options = config.retrieval.rerankingOptions
            )
        } else {
            documents
        }
        
        // Build context from retrieved documents
        val context = buildContext(rerankedDocuments, question, options)
        
        // Build prompt with context and question
        val prompt = buildPrompt(question, context)
        
        // Generate answer from LLM
        val answer = llmClient.generate(prompt)
        
        val processingTime = System.currentTimeMillis() - startTime
        
        return RAGResponse(
            answer = answer,
            documents = rerankedDocuments,
            processingTimeMs = processingTime,
            metadata = mapOf(
                "questionEmbeddingDimensions" to questionEmbedding.size,
                "documentsRetrieved" to documents.size,
                "contextLength" to context.length,
                "reranked" to (config.retrieval.reranking || options?.rerank == true)
            )
        )
    }
    
    /**
     * Builds the context from retrieved documents
     */
    private fun buildContext(documents: List<ScoredDocument>, question: String, options: QueryOptions?): String {
        // Check if we should include metadata
        val includeMetadata = options?.includeMetadata ?: config.generation.includeMetadata
        
        return documents.joinToString("\n\n") { scoredDoc ->
            val doc = scoredDoc.document
            val content = doc.content
            
            if (includeMetadata && doc.metadata.isNotEmpty()) {
                // Format metadata according to template
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
     * Index text content directly
     * 
     * @param content Text content to index
     * @param id Optional document ID (generated if not provided)
     * @param metadata Optional metadata
     * @param chunkContent Whether to chunk the content before indexing
     * @return Boolean indicating success
     */
    override suspend fun indexFromText(content: String, id: String?, metadata: Map<String, Any>): Boolean {
        return indexFromText(content, id, metadata, true)
    }

    suspend fun indexFromText(content: String, id: String?, metadata: Map<String, Any>, chunkContent: Boolean): Boolean {
        // Preprocess the text if enabled
        val processedContent = if (config.indexing.preprocessText) {
            TextPreprocessor(config.indexing.textPreprocessingConfig).preprocess(content)
        } else {
            content
        }
        
        val docId = id ?: generateDocumentId(processedContent)
        val document = SimpleDocument(docId, processedContent, metadata)
        
        return if (chunkContent) {
            indexDocumentWithChunking(document)
        } else {
            indexDocument(document)
        }
    }
    
    /**
     * Index a document after chunking it
     * 
     * @param document The document to chunk and index
     * @return Boolean indicating success
     */
    override suspend fun indexDocumentWithChunking(document: Document): Boolean {
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
            val chunks = chunker.chunkDocument(document, config.indexing.chunkingStrategy, chunkingConfig)
            
            // Index all chunks
            return indexDocuments(chunks).all { it }
        } catch (e: Exception) {
            // In a real implementation, we would log this error
            return false
        }
    }
    
    /**
     * Generate a document ID based on content
     */
    private fun generateDocumentId(content: String): String {
        val hash = content.hashCode().absoluteValue.toString(16)
        return "doc-$hash"
    }
    
    /**
     * Index content from a file
     * 
     * @param filePath Path to the file to index
     * @param metadata Optional metadata
     * @param chunkContent Whether to chunk the content before indexing
     * @return Boolean indicating success
     */
    override suspend fun indexFromFile(filePath: String, metadata: Map<String, Any>): Boolean {
        return indexFromFile(filePath, metadata, true)
    }

    suspend fun indexFromFile(filePath: String, metadata: Map<String, Any>, chunkContent: Boolean): Boolean {
        try {
            val file = File(filePath)
            if (!file.exists() || !file.isFile) {
                return false
            }
            
            // Use the appropriate extractor based on file extension
            val extractor = DocumentExtractorFactory.getExtractorForFile(file.name)
            val document = extractor.extract(file.inputStream(), metadata)
            
            return if (chunkContent) {
                indexDocumentWithChunking(document)
            } else {
                indexDocument(document)
            }
        } catch (e: Exception) {
            // In a real implementation, we would log this error
            return false
        }
    }
    
    // Get or create a reranker based on configuration
    private fun getReranker(): Reranker {
        return reranker ?: when (config.retrieval.rerankingStrategy) {
            RerankingStrategy.MMR -> MaximalMarginalRelevanceReranker()
            RerankingStrategy.CROSS_ENCODER -> CrossEncoderReranker()
            RerankingStrategy.ENSEMBLE -> {
                // Create an ensemble with MMR reranker and a weight of 1.0
                val mmrReranker = MaximalMarginalRelevanceReranker()
                EnsembleReranker(listOf(mmrReranker to 1.0f))
            }
            RerankingStrategy.HIERARCHICAL -> {
                // Hierarchical reranking (default to MMR for now)
                // TODO: Implement proper hierarchical reranking
                MaximalMarginalRelevanceReranker()
            }
            RerankingStrategy.NONE -> {
                // No-op reranker that just returns the documents as-is
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
    
    // Get or create a chunking manager
    private fun getChunkingManager(): ChunkingManager {
        return chunkingManager ?: ChunkingManager(embedder)
    }
}

/**
 * Builder for RAG instances
 */
class RAGBuilder {
    var embedder: Embedder? = null
    var vectorStore: VectorStore? = null
    var llmClient: LLMClient? = null
    var config = RAGConfig()
    var reranker: Reranker? = null
    var chunkingManager: ChunkingManager? = null
    
    /**
     * Configure the RAG instance
     */
    fun config(init: RAGConfig.() -> Unit) {
        config = RAGConfig().apply(init)
    }
    
    /**
     * Configure indexing
     */
    fun indexing(init: com.gazapps.rag.core.config.IndexingConfig.() -> Unit) {
        config.indexing.apply(init)
    }
    
    /**
     * Configure retrieval
     */
    fun retrieval(init: com.gazapps.rag.core.config.RetrievalConfig.() -> Unit) {
        config.retrieval.apply(init)
    }
    
    /**
     * Configure generation
     */
    fun generation(init: com.gazapps.rag.core.config.GenerationConfig.() -> Unit) {
        config.generation.apply(init)
    }
    
    /**
     * Configure the embedding model
     */
    fun withEmbedder(embedder: Embedder): RAGBuilder {
        this.embedder = embedder
        return this
    }
    
    /**
     * Configure the vector store
     */
    fun withVectorStore(vectorStore: VectorStore): RAGBuilder {
        this.vectorStore = vectorStore
        return this
    }
    
    /**
     * Configure the LLM client
     */
    fun withLLMClient(llmClient: LLMClient): RAGBuilder {
        this.llmClient = llmClient
        return this
    }
    
    /**
     * Configure the reranker
     */
    fun withReranker(reranker: Reranker): RAGBuilder {
        this.reranker = reranker
        return this
    }
    
    /**
     * Configure the chunking manager
     */
    fun withChunkingManager(chunkingManager: ChunkingManager): RAGBuilder {
        this.chunkingManager = chunkingManager
        return this
    }
    
    /**
     * Build the RAG instance
     */
    fun build(): RAG {
        val actualEmbedder = embedder ?: throw IllegalStateException("Embedder must be provided")
        val actualVectorStore = vectorStore ?: throw IllegalStateException("VectorStore must be provided")
        val actualLLMClient = llmClient ?: throw IllegalStateException("LLMClient must be provided")
        
        return RAG(
            embedder = actualEmbedder,
            vectorStore = actualVectorStore,
            llmClient = actualLLMClient,
            config = config,
            reranker = reranker,
            chunkingManager = chunkingManager
        )
    }
}

/**
 * DSL function to create RAG instances
 */
fun rag(init: RAGBuilder.() -> Unit): RAG {
    return RAGBuilder().apply(init).build()
}

/**
 * Alternative Builder pattern with method chaining
 */
fun ragBuilder(): RAGBuilder = RAGBuilder()
