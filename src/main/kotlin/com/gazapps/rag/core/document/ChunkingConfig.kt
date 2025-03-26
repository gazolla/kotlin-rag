package com.gazapps.rag.core.document

/**
 * Configuração para o processo de chunking de documentos.
 */
data class ChunkingConfig(
    val chunkSize: Int = 500,
    val chunkOverlap: Int = 50,
    val strategy: ChunkingStrategy = ChunkingStrategy.PARAGRAPH,
    val preserveMetadata: Boolean = true,
    val includeChunkMetadata: Boolean = true,
    val maxChunkSize: Int = 500,
    val overlap: Int = 50
)

/**
 * Estratégias disponíveis para chunking de texto.
 */
enum class ChunkingStrategy {
    SENTENCE, PARAGRAPH, SECTION, FIXED_SIZE, SEMANTIC, HIERARCHICAL
}
