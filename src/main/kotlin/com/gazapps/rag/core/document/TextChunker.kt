package com.gazapps.rag.core.document

import com.gazapps.rag.core.Document
import com.gazapps.rag.core.SimpleDocument

/**
 * Classe responsável por dividir textos em chunks menores.
 */
class TextChunker(
    val config: ChunkingConfig = ChunkingConfig()
) {
    /**
     * Divide um texto em chunks menores de acordo com a estratégia configurada.
     */
    fun chunk(text: String): List<String> {
        return when (config.strategy) {
            ChunkingStrategy.PARAGRAPH -> chunkByParagraphs(text)
            ChunkingStrategy.SENTENCE -> chunkBySentences(text)
            ChunkingStrategy.SECTION -> chunkBySections(text)
            ChunkingStrategy.FIXED_SIZE -> chunkByFixedSize(text)
            ChunkingStrategy.SEMANTIC -> chunkBySemantic(text)
            ChunkingStrategy.HIERARCHICAL -> chunkByHierarchical(text)
        }
    }

    /**
     * Divide um documento em múltiplos documentos menores.
     */
    fun chunkDocument(document: Document): List<Document> {
        val chunks = chunk(document.content)
        return chunks.mapIndexed { index, content ->
            SimpleDocument(
                id = "${document.id}_chunk_$index",
                content = content,
                metadata = document.metadata + mapOf(
                    "chunk_index" to index,
                    "parent_id" to document.id,
                    "chunk_count" to chunks.size
                )
            )
        }
    }

    private fun chunkByParagraphs(text: String): List<String> {
        val paragraphs = text.split(Regex("\\n\\s*\\n"))
            .map { it.trim() }
            .filter { it.isNotEmpty() }

        return createChunksFromUnits(paragraphs)
    }

    private fun chunkBySentences(text: String): List<String> {
        val sentences = text.split(Regex("(?<=[.!?])\\s+"))
            .map { it.trim() }
            .filter { it.isNotEmpty() }

        return createChunksFromUnits(sentences)
    }

    private fun chunkBySections(text: String): List<String> {
        val sections = text.split(Regex("(?m)^(#{1,6}\\s|<h[1-6]>).*?$"))
            .map { it.trim() }
            .filter { it.isNotEmpty() }

        return createChunksFromUnits(sections)
    }

    private fun chunkByFixedSize(text: String): List<String> {
        val chunks = mutableListOf<String>()
        var i = 0
        
        while (i < text.length) {
            val endIndex = minOf(i + config.chunkSize, text.length)
            chunks.add(text.substring(i, endIndex))
            i += config.chunkSize - config.chunkOverlap
        }
        
        return chunks
    }

    private fun chunkBySemantic(text: String): List<String> {
        // Implementação simplificada - em produção usaria análise semântica
        return chunkByParagraphs(text)
    }
    
    private fun chunkByHierarchical(text: String): List<String> {
        // Implementação do chunking hierárquico
        val sections = text.split(Regex("(?m)^#+\\s+"))
            .filter { it.isNotEmpty() }
        
        val result = mutableListOf<String>()
        
        for (section in sections) {
            val paragraphs = section.split("\n\n")
            var currentChunk = StringBuilder()
            
            for (paragraph in paragraphs) {
                if (currentChunk.length + paragraph.length > config.chunkSize && currentChunk.isNotEmpty()) {
                    result.add(currentChunk.toString())
                    currentChunk = StringBuilder()
                    
                    // Adicionar sobreposição
                    val overlapSize = config.chunkOverlap.coerceAtMost(paragraph.length)
                    if (overlapSize > 0) {
                        currentChunk.append(paragraph.substring(0, overlapSize))
                    }
                }
                
                if (currentChunk.isNotEmpty()) {
                    currentChunk.append("\n\n")
                }
                currentChunk.append(paragraph)
            }
            
            if (currentChunk.isNotEmpty()) {
                result.add(currentChunk.toString())
            }
        }
        
        return result
    }

    private fun createChunksFromUnits(units: List<String>): List<String> {
        val chunks = mutableListOf<String>()
        var currentChunk = StringBuilder()
        
        for (unit in units) {
            if (currentChunk.length + unit.length > config.chunkSize && currentChunk.isNotEmpty()) {
                chunks.add(currentChunk.toString())
                currentChunk = StringBuilder()
                
                // Adicionar sobreposição se necessário
                if (config.chunkOverlap > 0 && chunks.size > 0) {
                    val lastChunk = chunks.last()
                    val overlapText = lastChunk.takeLast(config.chunkOverlap)
                    if (overlapText.isNotEmpty()) {
                        currentChunk.append(overlapText).append("\n\n")
                    }
                }
            }
            
            if (currentChunk.isNotEmpty()) {
                currentChunk.append("\n\n")
            }
            currentChunk.append(unit)
        }
        
        if (currentChunk.isNotEmpty()) {
            chunks.add(currentChunk.toString())
        }
        
        return chunks
    }
}
