package com.gazapps.rag.core

/**
 * Interface for vector stores that store and retrieve document embeddings.
 * 
 * Vector stores are specialized databases that index and search high-dimensional vectors
 * to efficiently retrieve documents based on semantic similarity.
 */
interface VectorStore {
    /**
     * Store a single document with its embedding vector.
     *
     * @param document The document to store
     * @param embedding The embedding vector representation of the document
     */
    suspend fun store(document: Document, embedding: FloatArray)
    
    /**
     * Store multiple documents with their corresponding embedding vectors.
     * This is typically more efficient than multiple individual calls.
     *
     * @param documents List of documents to store
     * @param embeddings List of embedding vectors corresponding to the documents
     */
    suspend fun batchStore(documents: List<Document>, embeddings: List<FloatArray>)
    
    /**
     * Search for documents similar to a query vector.
     *
     * @param query The query embedding vector
     * @param limit Maximum number of results to return
     * @param filter Optional metadata filter to restrict results
     * @return List of documents with similarity scores, ordered by decreasing similarity
     */
    suspend fun search(query: FloatArray, limit: Int = 5, filter: Map<String, Any>? = null): List<ScoredDocument>
    
    /**
     * Delete a document from the vector store.
     *
     * @param documentId ID of the document to delete
     */
    suspend fun delete(documentId: String)
    
    /**
     * Clear all documents from the vector store.
     */
    suspend fun clear()
}
