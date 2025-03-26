package com.gazapps.rag.core.document

import com.gazapps.rag.core.Document
import com.gazapps.rag.core.SimpleDocument
import com.gazapps.rag.core.Embedder
import com.gazapps.rag.core.VectorUtils
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.slf4j.LoggerFactory

/**
 * Advanced chunking strategies beyond basic text splitting
 */
enum class AdvancedChunkingStrategy {
    SLIDING_WINDOW,  // Overlapping windows with configurable stride
    HYBRID,          // Hybrid approach that chunks by semantics and then applies fixed-size chunking
    CONTENT_AWARE,   // Content-aware chunking that recognizes semantic boundaries
    HIERARCHICAL,    // Hierarchical chunking that maintains document structure
}

/**
 * Configuration for advanced chunking strategies
 */
data class AdvancedChunkingConfig(
    val strategy: AdvancedChunkingStrategy = AdvancedChunkingStrategy.SLIDING_WINDOW,
    val windowSize: Int = 500,
    val stride: Int = 250,
    val minChunkSize: Int = 100,
    val preserveMetadata: Boolean = true,
    val detectHeaders: Boolean = true,
    val hierarchyLevels: Int = 2,
    val includeChunkMetadata: Boolean = true
)

/**
 * Advanced chunker that implements sophisticated chunking strategies
 */
class AdvancedChunker(
    private val config: AdvancedChunkingConfig = AdvancedChunkingConfig()
) {
    private val logger = LoggerFactory.getLogger(AdvancedChunker::class.java)
    
    /**
     * Chunk a document using the configured strategy
     */
    fun chunkDocument(document: Document): List<Document> {
        return when (config.strategy) {
            AdvancedChunkingStrategy.SLIDING_WINDOW -> slidingWindowChunk(document)
            AdvancedChunkingStrategy.HYBRID -> hybridChunk(document)
            AdvancedChunkingStrategy.CONTENT_AWARE -> contentAwareChunk(document)
            AdvancedChunkingStrategy.HIERARCHICAL -> hierarchicalChunk(document)
        }
    }
    
    /**
     * Sliding window chunking with configurable stride
     */
    private fun slidingWindowChunk(document: Document): List<Document> {
        val text = document.content
        val chunks = mutableListOf<String>()
        var currentPos = 0
        
        // Calculate the window size and stride in terms of characters
        val windowSize = config.windowSize
        val stride = config.stride
        
        while (currentPos < text.length) {
            val end = minOf(currentPos + windowSize, text.length)
            
            // Extract the chunk
            val chunk = text.substring(currentPos, end)
            if (chunk.isNotEmpty()) {
                chunks.add(chunk)
            }
            
            // Move by the stride amount
            currentPos += stride
        }
        
        logger.debug("Sliding window chunking created ${chunks.size} chunks from document")
        
        return chunks.mapIndexed { index, content ->
            createChunkDocument(document, content, index, chunks.size)
        }
    }
    
    /**
     * Hybrid chunking that uses semantic boundaries and then applies fixed-size chunking
     */
    private fun hybridChunk(document: Document): List<Document> {
        // First, try to identify semantic sections
        val text = document.content
        val sections = identifySections(text)
        
        if (sections.isEmpty()) {
            // Fallback to sliding window if no sections identified
            return slidingWindowChunk(document)
        }
        
        val finalChunks = mutableListOf<Document>()
        var chunkIndex = 0
        
        for ((sectionIndex, section) in sections.withIndex()) {
            if (section.length <= config.windowSize) {
                // Section is small enough, use as is
                finalChunks.add(createChunkDocument(
                    document, 
                    section, 
                    chunkIndex++, 
                    sections.size, 
                    mapOf("section_index" to sectionIndex)
                ))
            } else {
                // Section is too large, apply sliding window to it
                var currentPos = 0
                while (currentPos < section.length) {
                    val end = minOf(currentPos + config.windowSize, section.length)
                    val chunk = section.substring(currentPos, end)
                    
                    finalChunks.add(createChunkDocument(
                        document, 
                        chunk, 
                        chunkIndex++, 
                        sections.size, 
                        mapOf(
                            "section_index" to sectionIndex,
                            "section_position" to currentPos,
                            "section_total_length" to section.length
                        )
                    ))
                    
                    currentPos += config.stride
                }
            }
        }
        
        logger.debug("Hybrid chunking created ${finalChunks.size} chunks from document")
        return finalChunks
    }
    
    /**
     * Content-aware chunking that recognizes semantic boundaries
     */
    private fun contentAwareChunk(document: Document): List<Document> {
        val text = document.content
        
        // Define patterns for semantic boundaries
        val boundaries = mutableListOf<Int>()
        boundaries.add(0) // Start of document
        
        // Add paragraph boundaries
        val paragraphs = text.split("\n\n")
        var currentPos = 0
        for (paragraph in paragraphs) {
            currentPos += paragraph.length
            boundaries.add(currentPos)
            currentPos += 2 // for the "\n\n"
        }
        
        // Add sentence boundaries within paragraphs
        val sentenceEnds = Regex("[.!?]\\s+").findAll(text)
        for (sentenceEnd in sentenceEnds) {
            boundaries.add(sentenceEnd.range.last + 1)
        }
        
        // Add header boundaries if enabled
        if (config.detectHeaders) {
            val headerPattern = Regex("(?m)^(#{1,6}\\s.+$|[A-Z][A-Za-z\\s]+\\n[-=]+$)")
            headerPattern.findAll(text).forEach {
                boundaries.add(it.range.first)
            }
        }
        
        // Sort and remove duplicates
        val sortedBoundaries = boundaries.distinct().sorted()
        
        // Create semantic chunks using the dynamic programming to find optimal chunking
        val chunks = findOptimalChunks(text, sortedBoundaries)
        
        logger.debug("Content-aware chunking created ${chunks.size} chunks from document")
        
        return chunks.mapIndexed { index, content ->
            createChunkDocument(document, content, index, chunks.size)
        }
    }
    
    /**
     * Hierarchical chunking that maintains document structure
     */
    private fun hierarchicalChunk(document: Document): List<Document> {
        val text = document.content
        
        // Identify top-level sections (e.g., headers)
        val sections = identifySections(text)
        
        // Process each section
        val allChunks = mutableListOf<Document>()
        var globalChunkIndex = 0
        
        for ((sectionIndex, section) in sections.withIndex()) {
            // Create a chunk for the section itself
            val sectionChunk = createChunkDocument(
                document,
                section,
                globalChunkIndex++,
                -1, // Will update later
                mapOf(
                    "hierarchy_level" to 0,
                    "section_index" to sectionIndex,
                    "is_section_header" to true
                )
            )
            allChunks.add(sectionChunk)
            
            // Break down section into paragraphs for level 1
            if (config.hierarchyLevels > 1) {
                val paragraphs = section.split("\n\n")
                
                for ((paraIndex, paragraph) in paragraphs.withIndex()) {
                    if (paragraph.length >= config.minChunkSize) {
                        val paragraphChunk = createChunkDocument(
                            document,
                            paragraph,
                            globalChunkIndex++,
                            -1,
                            mapOf(
                                "hierarchy_level" to 1,
                                "section_index" to sectionIndex,
                                "paragraph_index" to paraIndex,
                                "parent_chunk_id" to sectionChunk.id
                            )
                        )
                        allChunks.add(paragraphChunk)
                        
                        // Further break down paragraphs into sentences for level 2
                        if (config.hierarchyLevels > 2 && paragraph.length > config.windowSize) {
                            val sentences = paragraph.split(Regex("(?<=[.!?])\\s+"))
                            var currentSentenceChunk = StringBuilder()
                            var sentenceChunkIndex = 0
                            
                            for (sentence in sentences) {
                                if (currentSentenceChunk.length + sentence.length > config.windowSize) {
                                    if (currentSentenceChunk.isNotEmpty()) {
                                        allChunks.add(createChunkDocument(
                                            document,
                                            currentSentenceChunk.toString(),
                                            globalChunkIndex++,
                                            -1,
                                            mapOf(
                                                "hierarchy_level" to 2,
                                                "section_index" to sectionIndex,
                                                "paragraph_index" to paraIndex,
                                                "sentence_chunk_index" to sentenceChunkIndex++,
                                                "parent_chunk_id" to paragraphChunk.id
                                            )
                                        ))
                                        currentSentenceChunk = StringBuilder()
                                    }
                                }
                                
                                if (currentSentenceChunk.isNotEmpty()) {
                                    currentSentenceChunk.append(" ")
                                }
                                currentSentenceChunk.append(sentence)
                            }
                            
                            // Add any remaining sentences
                            if (currentSentenceChunk.isNotEmpty()) {
                                allChunks.add(createChunkDocument(
                                    document,
                                    currentSentenceChunk.toString(),
                                    globalChunkIndex++,
                                    -1,
                                    mapOf(
                                        "hierarchy_level" to 2,
                                        "section_index" to sectionIndex,
                                        "paragraph_index" to paraIndex,
                                        "sentence_chunk_index" to sentenceChunkIndex,
                                        "parent_chunk_id" to paragraphChunk.id
                                    )
                                ))
                            }
                        }
                    }
                }
            }
        }
        
        // Update total chunk count in metadata
        val totalChunks = allChunks.size
        val updatedChunks = allChunks.map { chunk ->
            if (chunk.metadata["chunk_count"] == -1) {
                val updatedMetadata = chunk.metadata.toMutableMap().apply {
                    this["chunk_count"] = totalChunks
                }
                SimpleDocument(chunk.id, chunk.content, updatedMetadata)
            } else {
                chunk
            }
        }
        
        logger.debug("Hierarchical chunking created $totalChunks chunks from document")
        return updatedChunks
    }
    
    /* ------- Helper Methods ------- */
    
    /**
     * Create a chunk document with the appropriate metadata
     */
    private fun createChunkDocument(
        originalDoc: Document,
        content: String,
        chunkIndex: Int,
        totalChunks: Int,
        additionalMetadata: Map<String, Any> = emptyMap()
    ): Document {
        val chunkId = "${originalDoc.id}-chunk-$chunkIndex"
        val metadata = if (config.preserveMetadata) {
            originalDoc.metadata.toMutableMap()
        } else {
            mutableMapOf()
        }
        
        if (config.includeChunkMetadata) {
            metadata["chunk_index"] = chunkIndex
            metadata["chunk_count"] = totalChunks
            metadata["parent_id"] = originalDoc.id
            metadata["chunk_strategy"] = config.strategy.name
            metadata.putAll(additionalMetadata)
        }
        
        return SimpleDocument(
            id = chunkId,
            content = content,
            metadata = metadata
        )
    }
    
    /**
     * Identify sections in the document based on headers or other semantic markers
     */
    private fun identifySections(text: String): List<String> {
        // Look for different types of headers
        val headerPattern = Regex("(?m)^(#{1,6}\\s.+$|[A-Z][A-Za-z\\s]+\\n[-=]+$)")
        
        val boundaries = mutableListOf<Int>()
        boundaries.add(0) // Start of document
        
        // Find all headers
        headerPattern.findAll(text).forEach {
            boundaries.add(it.range.first)
        }
        
        // Add end of text
        boundaries.add(text.length)
        
        // Extract sections using the boundaries
        val sections = mutableListOf<String>()
        for (i in 0 until boundaries.size - 1) {
            val section = text.substring(boundaries[i], boundaries[i + 1]).trim()
            if (section.isNotEmpty()) {
                sections.add(section)
            }
        }
        
        return if (sections.isEmpty()) {
            // Fallback to paragraphs if no headers found
            text.split("\n\n").filter { it.trim().isNotEmpty() }
        } else {
            sections
        }
    }
    
    /**
     * Find optimal chunks using dynamic programming
     * This tries to create chunks that respect semantic boundaries while
     * staying close to the desired chunk size
     */
    private fun findOptimalChunks(text: String, boundaries: List<Int>): List<String> {
        if (boundaries.isEmpty()) {
            return listOf(text)
        }
        
        // Define cost function
        fun cost(start: Int, end: Int): Float {
            val size = end - start
            // Penalize chunks that are too small or too large
                return when {
                    size < config.minChunkSize -> Float.MAX_VALUE
                    size > config.windowSize -> (size - config.windowSize).toFloat().pow(2f)
                    else -> (config.windowSize - size).toFloat()
                }
        }
        
        // Dynamic programming
        val n = boundaries.size
        val dp = FloatArray(n) { Float.MAX_VALUE }
        val parent = IntArray(n) { -1 }
        
        dp[0] = 0f
        
        for (i in 1 until n) {
            for (j in 0 until i) {
                val c = cost(boundaries[j], boundaries[i])
                if (dp[j] + c < dp[i]) {
                    dp[i] = dp[j] + c
                    parent[i] = j
                }
            }
        }
        
        // Reconstruct solution
        val chunks = mutableListOf<String>()
        var i = n - 1
        val indices = mutableListOf<Int>()
        
        while (i > 0) {
            indices.add(i)
            i = parent[i]
        }
        
        indices.reverse()
        var start = 0
        
        for (idx in indices) {
            val end = boundaries[idx]
            chunks.add(text.substring(start, end))
            start = end
        }
        
        return chunks
    }
    
    private fun Float.pow(exponent: Float): Float = Math.pow(this.toDouble(), exponent.toDouble()).toFloat()
}

/**
 * Class for semantic chunking that uses embeddings to find natural chunk boundaries
 */
class SemanticChunker(
    private val embedder: Embedder,
    private val windowSize: Int = 500,
    private val similarityThreshold: Float = 0.7f,
    private val minChunkSize: Int = 100
) {
    private val logger = LoggerFactory.getLogger(SemanticChunker::class.java)
    
    /**
     * Chunk a document based on semantic similarity of adjacent text blocks
     */
    suspend fun chunkDocument(document: Document): List<Document> {
        val text = document.content
        
        // Initial chunking by paragraphs
        val paragraphs = text.split("\n\n")
            .map { it.trim() }
            .filter { it.isNotEmpty() }
        
        if (paragraphs.size <= 1) {
            // Not enough paragraphs, fall back to fixed-size chunking
            logger.debug("Document has only one paragraph, falling back to fixed-size chunking")
            return fallbackChunk(document)
        }
        
        // Get embeddings for each paragraph
        val embeddings = withContext(Dispatchers.Default) {
            embedder.batchEmbed(paragraphs)
        }
        
        // Find semantic boundaries
        val boundaries = mutableListOf<Int>()
        boundaries.add(0) // Start with the first paragraph
        
        for (i in 0 until paragraphs.size - 1) {
            val similarity = VectorUtils.cosineSimilarity(embeddings[i], embeddings[i + 1])
            if (similarity < similarityThreshold) {
                // Found a semantic boundary
                boundaries.add(i + 1)
            }
        }
        
        if (boundaries.size <= 1) {
            // No clear semantic boundaries found
            logger.debug("No clear semantic boundaries found, falling back to fixed-size chunking")
            return fallbackChunk(document)
        }
        
        // Add the end boundary
        boundaries.add(paragraphs.size)
        
        // Create chunks based on boundaries
        val chunks = mutableListOf<Document>()
        for (i in 0 until boundaries.size - 1) {
            val startIdx = boundaries[i]
            val endIdx = boundaries[i + 1]
            
            // Combine paragraphs into a chunk
            val chunkParagraphs = paragraphs.subList(startIdx, endIdx)
            val chunkText = chunkParagraphs.joinToString("\n\n")
            
            // If chunk is too large, further split it
            if (chunkText.length > windowSize) {
                // Further split this chunk
                val subChunks = chunkText.chunked(windowSize)
                subChunks.forEachIndexed { subIdx, subChunk ->
                    val chunkId = "${document.id}-chunk-${chunks.size}"
                    chunks.add(SimpleDocument(
                        id = chunkId,
                        content = subChunk,
                        metadata = document.metadata.toMutableMap().apply {
                            this["chunk_index"] = chunks.size
                            this["parent_id"] = document.id
                            this["semantic_boundary_start"] = startIdx
                            this["semantic_boundary_end"] = endIdx
                            this["sub_chunk_index"] = subIdx
                        }
                    ))
                }
            } else {
                // Add the chunk as is
                val chunkId = "${document.id}-chunk-${chunks.size}"
                chunks.add(SimpleDocument(
                    id = chunkId,
                    content = chunkText,
                    metadata = document.metadata.toMutableMap().apply {
                        this["chunk_index"] = chunks.size
                        this["parent_id"] = document.id
                        this["semantic_boundary_start"] = startIdx
                        this["semantic_boundary_end"] = endIdx
                    }
                ))
            }
        }
        
        // Update chunk metadata with total count
        return chunks.mapIndexed { index, chunk ->
            SimpleDocument(
                id = chunk.id,
                content = chunk.content,
                metadata = chunk.metadata.toMutableMap().apply {
                    this["chunk_count"] = chunks.size
                }
            )
        }
    }
    
    /**
     * Fallback to fixed-size chunking
     */
    private fun fallbackChunk(document: Document): List<Document> {
        val text = document.content
        val chunks = mutableListOf<String>()
        var currentPos = 0
        
        while (currentPos < text.length) {
            val end = minOf(currentPos + windowSize, text.length)
            chunks.add(text.substring(currentPos, end))
            currentPos += windowSize
        }
        
        return chunks.mapIndexed { index, content ->
            SimpleDocument(
                id = "${document.id}-chunk-$index",
                content = content,
                metadata = document.metadata.toMutableMap().apply {
                    this["chunk_index"] = index
                    this["chunk_count"] = chunks.size
                    this["parent_id"] = document.id
                    this["fallback_chunking"] = true
                }
            )
        }
    }
}
