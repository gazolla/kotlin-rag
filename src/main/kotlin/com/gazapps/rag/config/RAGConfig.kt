package com.gazapps.rag.config

import com.gazapps.rag.core.error.LogLevel

/**
 * Configuração para o sistema RAG.
 * 
 * Esta classe deve ser usada para compatibilidade com o pacote com.gazapps.rag.core.config.RAGConfig
 * @deprecated Use com.gazapps.rag.core.config.RAGConfig em vez desta classe
 */
@Deprecated(
    message = "Use com.gazapps.rag.core.config.RAGConfig em vez desta classe",
    replaceWith = ReplaceWith("com.gazapps.rag.core.config.RAGConfig"),
    level = DeprecationLevel.WARNING
)
data class RAGConfig(
    /**
     * Tamanho de cada chunk em tokens.
     */
    val chunkSize: Int = 500,
    
    /**
     * Sobreposição entre chunks em tokens.
     */
    val chunkOverlap: Int = 50,
    
    /**
     * Número máximo de documentos a serem recuperados em consultas.
     */
    val retrievalLimit: Int = 5,
    
    /**
     * Template para o prompt enviado ao LLM.
     */
    val promptTemplate: String = """
        Com base no contexto a seguir, responda à pergunta.
        
        Contexto:
        {context}
        
        Pergunta:
        {question}
        
        Resposta:
    """.trimIndent(),
    
    /**
     * Limite mínimo de similaridade para considerar um documento relevante.
     */
    val similarityThreshold: Float = 0.7f,
    
    /**
     * Se deve incluir metadados dos documentos no contexto.
     */
    val includeMetadata: Boolean = true,
    
    /**
     * Se o cache de embeddings está ativado.
     */
    val cacheEnabled: Boolean = true,
    
    /**
     * Nível de log do sistema.
     */
    val logLevel: LogLevel = LogLevel.INFO,
    
    /**
     * Número máximo de tentativas para operações com fallback.
     */
    val maxRetries: Int = 3
)

/**
 * Converte a configuração obsoleta para a configuração core.
 * 
 * @return Instância de com.gazapps.rag.core.config.RAGConfig com valores correspondentes
 */
fun com.gazapps.rag.config.RAGConfig.toCore(): com.gazapps.rag.core.config.RAGConfig {
    return com.gazapps.rag.core.config.RAGConfig().apply {
        this.chunkSize = this@toCore.chunkSize
        this.chunkOverlap = this@toCore.chunkOverlap
        this.retrievalLimit = this@toCore.retrievalLimit
        this.promptTemplate = this@toCore.promptTemplate
        this.similarityThreshold = this@toCore.similarityThreshold
        this.includeMetadata = this@toCore.includeMetadata
        this.cacheEnabled = this@toCore.cacheEnabled
        // Observe que logLevel e maxRetries podem precisar de tratamento especial
        // dependendo da implementação em core.config.RAGConfig
    }
}
