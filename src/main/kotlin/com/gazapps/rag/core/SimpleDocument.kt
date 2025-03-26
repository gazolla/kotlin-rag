package com.gazapps.rag.core

/**
 * Implementação simples da interface Document.
 * 
 * Representa um documento com id, conteúdo e metadados.
 */
data class SimpleDocument(
    override val id: String,
    override val content: String,
    override val metadata: Map<String, Any> = emptyMap(),
    override val chunks: List<Document>? = null
) : Document
