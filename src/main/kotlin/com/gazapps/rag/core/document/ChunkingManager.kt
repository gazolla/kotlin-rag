package com.gazapps.rag.core.document

import com.gazapps.rag.core.Document
import com.gazapps.rag.core.Embedder

/**
 * Gerencia o processo de divisão de documentos em chunks.
 */
class ChunkingManager(
    private val embedder: Embedder
) {
    /**
     * Divide um documento em múltiplos chunks menores.
     *
     * @param document O documento a ser dividido.
     * @param strategy Estratégia de chunking a ser utilizada.
     * @param config Configuração de chunking.
     * @return Lista de documentos representando os chunks.
     */
    fun chunkDocument(document: Document, strategy: ChunkingStrategy, config: ChunkingConfig): List<Document> {
        val chunker = TextChunker(config)
        return chunker.chunkDocument(document)
    }

    /**
     * Divide um documento em chunks e opcionalmente gera embeddings para cada chunk.
     *
     * @param document O documento a ser dividido.
     * @param strategy Estratégia de chunking a ser utilizada.
     * @param config Configuração de chunking.
     * @param generateEmbeddings Se verdadeiro, gera embeddings para cada chunk.
     * @return Lista de documentos representando os chunks.
     */
    suspend fun chunkAndEmbed(document: Document, strategy: ChunkingStrategy, config: ChunkingConfig, generateEmbeddings: Boolean = false): List<Pair<Document, FloatArray?>> {
        val chunks = chunkDocument(document, strategy, config)
        
        if (!generateEmbeddings) {
            return chunks.map { it to null }
        }
        
        val embeddings = embedder.batchEmbed(chunks.map { it.content })
        return chunks.zip(embeddings)
    }
}
