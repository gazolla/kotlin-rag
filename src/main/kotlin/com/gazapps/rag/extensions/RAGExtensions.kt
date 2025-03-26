package com.gazapps.rag.extensions

import com.gazapps.rag.core.*
import com.gazapps.rag.core.document.DocumentExtractorFactory
import com.gazapps.rag.core.SimpleDocument
import com.gazapps.rag.core.document.TextChunker
import com.gazapps.rag.core.document.ChunkingStrategy
import com.gazapps.rag.core.document.ChunkingConfig
import com.gazapps.rag.core.document.DocumentExtractor
import java.io.File
import java.io.InputStream
import java.net.URL
import kotlin.io.path.Path
import kotlin.io.path.readText

/**
 * Extension functions for the IRAG interface
 */

/**
 * Index content from a string directly
 * 
 * @param content The text content to index
 * @param id Optional document ID (defaults to hash of content)
 * @param metadata Optional metadata to attach to the document
 * @return Result indicating success or failure with error information
 */
suspend fun IRAG.indexText(content: String, id: String? = null, metadata: Map<String, Any> = emptyMap()): Result<Boolean> {
    return runCatching {
        indexFromText(content, id, metadata)
    }
}

/**
 * Index content from a file using its path
 * 
 * @param path Path to the file as a string
 * @param metadata Optional metadata to attach
 * @return Result indicating success or failure with error information
 */
suspend fun IRAG.indexFile(path: String, metadata: Map<String, Any> = emptyMap()): Result<Boolean> {
    return runCatching {
        indexFromFile(path, metadata)
    }
}

/**
 * Index content from a File object
 * 
 * @param file The file to index
 * @param metadata Optional metadata to attach
 * @return Result indicating success or failure with error information
 */
suspend fun IRAG.indexFile(file: File, metadata: Map<String, Any> = emptyMap()): Result<Boolean> {
    return runCatching {
        if (!file.exists() || !file.isFile) {
            throw IllegalArgumentException("File does not exist or is not a regular file: ${file.path}")
        }
        
        val fileMetadata = mapOf(
            "filename" to file.name,
            "filesize" to file.length(),
            "filepath" to file.absolutePath,
            "lastModified" to file.lastModified()
        ) + metadata
        
        val extractor = DocumentExtractorFactory.getExtractorForFile(file.name)
        val document = file.inputStream().use { stream -> extractor.extract(stream, fileMetadata) }
        indexDocumentWithChunking(document)
    }
}

/**
 * Index content from a URL
 * 
 * @param url URL to fetch and index content from
 * @param metadata Optional metadata to attach
 * @return Result indicating success or failure with error information
 */
suspend fun IRAG.indexUrl(url: String, metadata: Map<String, Any> = emptyMap()): Result<Boolean> {
    return runCatching {
        val urlObj = URL(url)
        val urlConnection = urlObj.openConnection()
        
        val contentType = urlConnection.contentType ?: "text/plain"
        val inputStream = urlConnection.getInputStream()
        
        val urlMetadata = mapOf(
            "source" to url,
            "content_type" to contentType,
            "indexed_at" to System.currentTimeMillis()
        ) + metadata
        
        // Determine extractor based on content type
        val extractor = DocumentExtractorFactory.getExtractor(contentType)
        
        val document = extractor.extract(inputStream, urlMetadata)
        indexDocumentWithChunking(document)
    }
}

/**
 * Ask a question and get the response with relevant documents
 * 
 * @param question The question to ask
 * @return Result containing RAGResponse or error
 */
suspend fun IRAG.ask(question: String): Result<RAGResponse> {
    return runCatching {
        query(question)
    }
}

/**
 * Ask a question with filtering
 * 
 * @param question The question to ask
 * @param filter Optional metadata filter
 * @return Result containing RAGResponse or error
 */
suspend fun IRAG.ask(question: String, filter: Map<String, Any>): Result<RAGResponse> {
    return runCatching {
        query(question, filter)
    }
}

/**
 * Extension functions for working with Files
 */

/**
 * Convert a File to a Document
 * 
 * @param id Optional document ID (defaults to filename)
 * @param metadata Additional metadata to attach
 * @return Document representing the file's content
 */
fun File.toDocument(id: String? = null, metadata: Map<String, Any> = emptyMap()): Document {
    require(exists() && isFile) { "File does not exist or is not a regular file: $path" }
    
    val fileMetadata = mapOf(
        "filename" to name,
        "filesize" to length(),
        "filepath" to absolutePath,
        "lastModified" to lastModified()
    ) + metadata
    
    val content = readText()
    return SimpleDocument(id ?: name, content, fileMetadata)
}

/**
 * Convert file content to chunks
 * 
 * @param strategy Chunking strategy to use
 * @param maxChunkSize Maximum size of each chunk
 * @param overlap Overlap between chunks
 * @return List of document chunks
 */
fun File.toChunks(
    strategy: ChunkingStrategy = ChunkingStrategy.PARAGRAPH,
    maxChunkSize: Int = 500,
    overlap: Int = 50
): List<Document> {
    require(exists() && isFile) { "File does not exist or is not a regular file: $path" }
    
    val content = readText()
    val baseDocument = toDocument()
    
    val chunker = TextChunker(ChunkingConfig(maxChunkSize, overlap, strategy))
    return chunker.chunkDocument(baseDocument)
}

/**
 * Extension functions for InputStreams
 */

/**
 * Convert an InputStream to a Document
 * 
 * @param id Document ID
 * @param metadata Additional metadata
 * @return Document representing the stream's content
 */
fun InputStream.toDocument(id: String, metadata: Map<String, Any> = emptyMap()): Document {
    val content = bufferedReader().use { it.readText() }
    return SimpleDocument(id, content, metadata)
}

/**
 * Extension method to extract a document from a file
 */
suspend fun File.extract(metadata: Map<String, Any> = emptyMap()): Document {
    val extension = extension.lowercase()
    val extractor = DocumentExtractorFactory.getExtractorForFile(name)
    return inputStream().use { stream -> extractor.extract(stream, metadata + mapOf("filename" to name)) }
}

/**
 * Extension functions for DocumentExtractor interface
 */

/**
 * Extract document from a File
 * 
 * @param file The file to extract from
 * @param metadata Additional metadata
 * @return Extracted document
 */
suspend fun DocumentExtractor.extract(file: File, metadata: Map<String, Any> = emptyMap()): Document {
    return file.inputStream().use { stream -> extract(stream, metadata + mapOf("filename" to file.name)) }
}
