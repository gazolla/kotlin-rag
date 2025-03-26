package com.gazapps.rag

import com.gazapps.rag.core.*
import com.gazapps.rag.core.config.RAGConfig
import com.gazapps.rag.core.document.ChunkingStrategy
import com.gazapps.rag.core.error.*
import com.gazapps.rag.core.monitoring.RAGMetricsManager
import com.gazapps.rag.extensions.*
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import java.io.File
import java.nio.file.Path
import kotlin.coroutines.CoroutineContext

/**
 * Main facade class for the Kotlin RAG library
 * Provides an ergonomic API for working with RAG functionality
 */
class KotlinRAG private constructor(
    private val ragImpl: IRAG,
    private val coroutineContext: CoroutineContext = Dispatchers.Default,
    private val config: RAGConfig = RAGConfig()
) {
    private val scope = CoroutineScope(coroutineContext)
    
    companion object {
        /**
         * Create a standard RAG instance with default settings
         */
        fun standard(
            embedder: Embedder,
            vectorStore: VectorStore,
            llmClient: LLMClient,
            config: RAGConfig = RAGConfig()
        ): KotlinRAG {
            val rag = rag {
                this.embedder = embedder
                this.vectorStore = vectorStore
                this.llmClient = llmClient
                this.config = config
            }
            
            return KotlinRAG(rag, Dispatchers.Default, config)
        }
        
        /**
         * Create a robust RAG instance with error handling and fallbacks
         */
        fun robust(
            embedder: Embedder,
            vectorStore: VectorStore,
            llmClient: LLMClient,
            fallbackEmbedder: Embedder? = null,
            fallbackVectorStore: VectorStore? = null,
            fallbackLLMClient: LLMClient? = null,
            config: RAGConfig = RAGConfig()
        ): KotlinRAG {
            val rag = ragWithErrorHandling {
                this.embedder = embedder
                this.vectorStore = vectorStore
                this.llmClient = llmClient
                this.fallbackEmbedder = fallbackEmbedder
                this.fallbackVectorStore = fallbackVectorStore
                this.fallbackLLMClient = fallbackLLMClient
                this.config = config
            }
            
            return KotlinRAG(rag, Dispatchers.Default, config)
        }
    }
    
    /**
     * Ask a question and get a response
     */
    suspend fun ask(question: String): RAGResponse {
        return ragImpl.query(question)
    }
    
    /**
     * Ask a question with filtering
     */
    suspend fun ask(question: String, filter: Map<String, Any>): RAGResponse {
        return ragImpl.query(question, filter)
    }
    
    /**
     * Ask a question with custom options
     */
    suspend fun ask(question: String, options: QueryOptions): RAGResponse {
        return ragImpl.query(question, options)
    }
    
    /**
     * Index text content
     */
    suspend fun indexText(content: String, id: String? = null, metadata: Map<String, Any> = emptyMap()): Boolean {
        return ragImpl.indexFromText(content, id, metadata)
    }
    
    /**
     * Index text content asynchronously
     */
    fun indexTextAsync(
        content: String, 
        id: String? = null, 
        metadata: Map<String, Any> = emptyMap(),
        onComplete: (Boolean) -> Unit = {}
    ) {
        scope.launch {
            val result = indexText(content, id, metadata)
            onComplete(result)
        }
    }
    
    /**
     * Index a document
     */
    suspend fun indexDocument(document: Document): Boolean {
        return ragImpl.indexDocument(document)
    }
    
    /**
     * Index a document asynchronously
     */
    fun indexDocumentAsync(document: Document, onComplete: (Boolean) -> Unit = {}) {
        scope.launch {
            val result = indexDocument(document)
            onComplete(result)
        }
    }
    
    /**
     * Index a file
     */
    suspend fun indexFile(file: File, metadata: Map<String, Any> = emptyMap()): Boolean {
        return ragImpl.indexFile(file, metadata).getOrElse { false }
    }
    
    /**
     * Index a file asynchronously
     */
    fun indexFileAsync(file: File, metadata: Map<String, Any> = emptyMap(), onComplete: (Boolean) -> Unit = {}) {
        scope.launch {
            val result = indexFile(file, metadata)
            onComplete(result)
        }
    }
    
    /**
     * Index a file from a path
     */
    suspend fun indexFile(path: String, metadata: Map<String, Any> = emptyMap()): Boolean {
        return ragImpl.indexFromFile(path, metadata)
    }
    
    /**
     * Index a file from a path asynchronously
     */
    fun indexFileAsync(path: String, metadata: Map<String, Any> = emptyMap(), onComplete: (Boolean) -> Unit = {}) {
        scope.launch {
            val result = indexFile(path, metadata)
            onComplete(result)
        }
    }
    
    /**
     * Index a URL
     */
    suspend fun indexUrl(url: String, metadata: Map<String, Any> = emptyMap()): Boolean {
        return ragImpl.indexUrl(url, metadata).getOrElse { false }
    }
    
    /**
     * Index a URL asynchronously
     */
    fun indexUrlAsync(url: String, metadata: Map<String, Any> = emptyMap(), onComplete: (Boolean) -> Unit = {}) {
        scope.launch {
            val result = indexUrl(url, metadata)
            onComplete(result)
        }
    }
    
    /**
     * Index a directory of files
     */
    suspend fun indexDirectory(
        directory: File,
        recursive: Boolean = true,
        fileExtensions: Set<String>? = null,
        metadata: Map<String, Any> = emptyMap()
    ): Map<String, Boolean> {
        require(directory.exists() && directory.isDirectory) {
            "Path does not exist or is not a directory: ${directory.absolutePath}"
        }
        
        val files = if (recursive) {
            directory.walkTopDown()
        } else {
            directory.listFiles()?.asSequence() ?: emptySequence()
        }
        
        val filesToIndex = files
            .filter { it.isFile }
            .filter { file ->
                fileExtensions?.any { ext -> file.name.endsWith(".$ext", ignoreCase = true) } ?: true
            }
            .toList()
        
        val results = mutableMapOf<String, Boolean>()
        
        for (file in filesToIndex) {
            val fileMetadata = metadata + mapOf(
                "relative_path" to file.relativeTo(directory).path,
                "directory" to directory.absolutePath
            )
            
            val success = indexFile(file, fileMetadata)
            results[file.absolutePath] = success
        }
        
        return results
    }
    
    /**
     * Index a directory of files asynchronously
     */
    fun indexDirectoryAsync(
        directory: File,
        recursive: Boolean = true,
        fileExtensions: Set<String>? = null,
        metadata: Map<String, Any> = emptyMap(),
        onProgress: (String, Boolean) -> Unit = { _, _ -> },
        onComplete: (Map<String, Boolean>) -> Unit = {}
    ) {
        scope.launch {
            val results = mutableMapOf<String, Boolean>()
            
            try {
                require(directory.exists() && directory.isDirectory) {
                    "Path does not exist or is not a directory: ${directory.absolutePath}"
                }
                
                val files = if (recursive) {
                    directory.walkTopDown()
                } else {
                    directory.listFiles()?.asSequence() ?: emptySequence()
                }
                
                val filesToIndex = files
                    .filter { it.isFile }
                    .filter { file ->
                        fileExtensions?.any { ext -> file.name.endsWith(".$ext", ignoreCase = true) } ?: true
                    }
                    .toList()
                
                for (file in filesToIndex) {
                    val fileMetadata = metadata + mapOf(
                        "relative_path" to file.relativeTo(directory).path,
                        "directory" to directory.absolutePath
                    )
                    
                    val success = indexFile(file, fileMetadata)
                    results[file.absolutePath] = success
                    
                    onProgress(file.absolutePath, success)
                }
            } catch (e: Exception) {
                // Log the error
            }
            
            onComplete(results)
        }
    }
    
    /**
     * Get the underlying RAG implementation
     */
    fun getRAGImplementation(): IRAG = ragImpl
}

/**
 * DSL function for creating a standard RAG instance
 */
fun kotlinRag(init: KotlinRAGBuilder.() -> Unit): KotlinRAG {
    val builder = KotlinRAGBuilder().apply(init)
    
    return if (builder.withErrorHandling) {
        KotlinRAG.robust(
            embedder = builder.embedder ?: throw IllegalStateException("Embedder must be provided"),
            vectorStore = builder.vectorStore ?: throw IllegalStateException("VectorStore must be provided"),
            llmClient = builder.llmClient ?: throw IllegalStateException("LLMClient must be provided"),
            fallbackEmbedder = builder.fallbackEmbedder,
            fallbackVectorStore = builder.fallbackVectorStore,
            fallbackLLMClient = builder.fallbackLLMClient,
            config = builder.config
        )
    } else {
        KotlinRAG.standard(
            embedder = builder.embedder ?: throw IllegalStateException("Embedder must be provided"),
            vectorStore = builder.vectorStore ?: throw IllegalStateException("VectorStore must be provided"),
            llmClient = builder.llmClient ?: throw IllegalStateException("LLMClient must be provided"),
            config = builder.config
        )
    }
}

/**
 * Builder for KotlinRAG
 */
class KotlinRAGBuilder {
    var embedder: Embedder? = null
    var vectorStore: VectorStore? = null
    var llmClient: LLMClient? = null
    var fallbackEmbedder: Embedder? = null
    var fallbackVectorStore: VectorStore? = null
    var fallbackLLMClient: LLMClient? = null
    var config = RAGConfig()
    var withErrorHandling: Boolean = false
    
    /**
     * Configure the embedder
     */
    fun embedder(embedder: Embedder) {
        this.embedder = embedder
    }
    
    /**
     * Configure the vector store
     */
    fun vectorStore(vectorStore: VectorStore) {
        this.vectorStore = vectorStore
    }
    
    /**
     * Configure the LLM client
     */
    fun llmClient(llmClient: LLMClient) {
        this.llmClient = llmClient
    }
    
    /**
     * Configure a fallback embedder
     */
    fun fallbackEmbedder(embedder: Embedder) {
        this.fallbackEmbedder = embedder
        this.withErrorHandling = true
    }
    
    /**
     * Configure a fallback vector store
     */
    fun fallbackVectorStore(vectorStore: VectorStore) {
        this.fallbackVectorStore = vectorStore
        this.withErrorHandling = true
    }
    
    /**
     * Configure a fallback LLM client
     */
    fun fallbackLLMClient(llmClient: LLMClient) {
        this.fallbackLLMClient = llmClient
        this.withErrorHandling = true
    }
    
    /**
     * Enable error handling
     */
    fun withErrorHandling() {
        this.withErrorHandling = true
    }
    
    /**
     * Configure the RAG instance
     */
    fun config(init: RAGConfig.() -> Unit) {
        config = RAGConfig().apply(init)
    }
}
