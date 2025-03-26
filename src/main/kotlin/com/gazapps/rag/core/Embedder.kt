package com.gazapps.rag.core

/**
 * Interface for embedding models that convert text to vector representations.
 * 
 * Embeddings are dense vector representations of text that capture semantic meaning,
 * allowing for similarity comparisons between texts.
 */
interface Embedder {
    /**
     * Convert a single text to an embedding vector.
     *
     * @param text The text to convert to an embedding
     * @return A floating-point vector representation of the input text
     */
    suspend fun embed(text: String): FloatArray
    
    /**
     * Convert multiple texts to embedding vectors in a single batch operation.
     * This is typically more efficient than multiple individual calls.
     *
     * @param texts List of texts to convert to embeddings
     * @return List of embedding vectors corresponding to the input texts
     */
    suspend fun batchEmbed(texts: List<String>): List<FloatArray>
}
