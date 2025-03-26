package com.gazapps.rag.core.document

/**
 * Main module export file with convenience functions for document processing components
 */

/**
 * Convenience function to chunk text with default settings
 */
fun chunkText(
    text: String, 
    maxSize: Int = 500, 
    strategy: ChunkingStrategy = ChunkingStrategy.PARAGRAPH
): List<String> {
    return TextChunker(
        ChunkingConfig(
            maxChunkSize = maxSize,
            strategy = strategy
        )
    ).chunk(text)
}

/**
 * Convenience function to preprocess text with default settings
 */
fun preprocessText(
    text: String,
    normalizeWhitespace: Boolean = true,
    removeHtml: Boolean = false,
    lowercase: Boolean = false
): String {
    return TextPreprocessor(
        PreprocessingConfig(
            normalizeWhitespace = normalizeWhitespace,
            removeHtml = removeHtml,
            lowercase = lowercase
        )
    ).preprocess(text)
}
