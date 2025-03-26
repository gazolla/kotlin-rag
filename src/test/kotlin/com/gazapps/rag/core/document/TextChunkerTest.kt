package com.gazapps.rag.core.document

import com.gazapps.rag.core.SimpleDocument
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class TextChunkerTest {

    private val sampleText = """
        # Introduction to RAG
        
        Retrieval-Augmented Generation (RAG) is a technique that combines retrieval of documents with generation of text.
        
        ## How RAG Works
        
        RAG works by first retrieving relevant documents from a knowledge base, then using them as context for a large language model.
        
        This helps the model ground its responses in factual information.
        
        ## Benefits of RAG
        
        1. Improved factuality
        2. Access to external knowledge
        3. Ability to cite sources
        4. Reduced hallucinations
        
        ## Implementation Challenges
        
        Building a RAG system requires solving several technical challenges:
        - Creating good vector embeddings
        - Storing and searching vectors efficiently
        - Chunking documents appropriately
        - Formulating effective prompts
    """.trimIndent()
    
    @Test
    fun `should chunk text by paragraphs`() {
        val chunker = TextChunker(ChunkingConfig(
            maxChunkSize = 200,
            strategy = ChunkingStrategy.PARAGRAPH
        ))
        
        val chunks = chunker.chunkText(sampleText)
        
        // Verify chunks were created
        assertTrue(chunks.isNotEmpty())
        
        // Each chunk should be under the max size
        chunks.forEach {
            assertTrue(it.length <= 200)
        }
        
        // First chunk should contain the title
        assertTrue(chunks.first().contains("Introduction to RAG"))
    }
    
    @Test
    fun `should chunk text by fixed size`() {
        val chunker = TextChunker(ChunkingConfig(
            maxChunkSize = 100,
            strategy = ChunkingStrategy.FIXED_SIZE
        ))
        
        val chunks = chunker.chunkText(sampleText)
        
        // Verify chunks were created
        assertTrue(chunks.isNotEmpty())
        
        // Most chunks should be close to max size (except possibly the last one)
        for (i in 0 until chunks.size - 1) {
            assertTrue(chunks[i].length <= 100)
            // Chunks should be reasonably sized (not tiny fragments)
            assertTrue(chunks[i].length >= 50)
        }
    }
    
    @Test
    fun `should chunk text by semantic structure`() {
        val chunker = TextChunker(ChunkingConfig(
            maxChunkSize = 300,
            strategy = ChunkingStrategy.SEMANTIC
        ))
        
        val chunks = chunker.chunkText(sampleText)
        
        // Verify chunks were created
        assertTrue(chunks.isNotEmpty())
        
        // Look for section headers in the chunks
        val hasIntroChunk = chunks.any { it.contains("Introduction to RAG") }
        val hasHowRagWorksChunk = chunks.any { it.contains("How RAG Works") }
        
        assertTrue(hasIntroChunk)
        assertTrue(hasHowRagWorksChunk)
    }
    
    @Test
    fun `should chunk document with metadata`() {
        val document = SimpleDocument(
            id = "doc-1",
            content = sampleText,
            metadata = mapOf("author" to "Tester", "date" to "2023-01-01")
        )
        
        val chunker = TextChunker(ChunkingConfig(
            maxChunkSize = 200,
            strategy = ChunkingStrategy.PARAGRAPH,
            preserveMetadata = true,
            includeChunkMetadata = true
        ))
        
        val chunks = chunker.chunkDocument(document)
        
        // Verify chunks were created
        assertTrue(chunks.isNotEmpty())
        
        // Check that original metadata is preserved
        chunks.forEach { chunk ->
            assertEquals("Tester", chunk.metadata["author"])
            assertEquals("2023-01-01", chunk.metadata["date"])
        }
        
        // Check that chunk metadata was added
        chunks.forEach { chunk ->
            assertTrue(chunk.metadata.containsKey("chunk_index"))
            assertEquals(document.id, chunk.metadata["parent_id"])
            assertEquals(chunks.size, chunk.metadata["chunk_count"])
        }
    }
}