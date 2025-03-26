package com.gazapps.rag.core

/**
 * Interface that defines the basic structure of a document.
 * 
 * A document is the fundamental unit of information in the RAG system,
 * containing text content and associated metadata.
 */
interface Document {
    /**
     * Unique identifier for the document.
     */
    val id: String
    
    /**
     * Text content of the document.
     */
    val content: String
    
    /**
     * Associated metadata for the document.
     * 
     * May include information such as source, author, date, URL, etc.
     */
    val metadata: Map<String, Any>
    
    /**
     * Optional list of chunks (sub-documents) that compose this document.
     * 
     * May be null if the document has not been chunked.
     */
    val chunks: List<Document>?
}

/**
 * Extension functions for Document
 */

/**
 * Create a new document by adding a metadata key-value pair
 * 
 * @param key Metadata key
 * @param value Metadata value
 * @return A new document with the additional metadata
 */
fun Document.withMetadata(key: String, value: Any): Document {
    return SimpleDocument(
        id = this.id,
        content = this.content,
        metadata = this.metadata + mapOf(key to value),
        chunks = this.chunks
    )
}

/**
 * Create a new document by adding multiple metadata key-value pairs
 * 
 * @param metadata Metadata to add
 * @return A new document with the additional metadata
 */
fun Document.withMetadata(metadata: Map<String, Any>): Document {
    return SimpleDocument(
        id = this.id,
        content = this.content,
        metadata = this.metadata + metadata,
        chunks = this.chunks
    )
}
