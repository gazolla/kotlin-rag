package com.gazapps.rag.core.config

/**
 * Configuração específica para o processo de geração de respostas.
 */
data class GenerationConfig(
    /**
     * Template para construção de prompts com contexto e pergunta.
     */
    var promptTemplate: String = "Based on the following context:\n\n{context}\n\nAnswer the question: {question}",
    
    /**
     * Se deve incluir metadados dos documentos no contexto.
     */
    var includeMetadata: Boolean = true,
    
    /**
     * Template para formatação de metadados dos documentos.
     */
    var metadataTemplate: String = "Source: {source}\nDate: {date}"
)
