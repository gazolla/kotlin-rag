package com.gazapps.rag.core

/**
 * Interface principal para Retrieval-Augmented Generation (RAG).
 * 
 * Define as operações principais para indexação de documentos e
 * realização de consultas RAG.
 */
interface IRAG {
    /**
     * Indexa um documento.
     * 
     * @param document O documento a ser indexado.
     * @return true se a indexação foi bem-sucedida, false caso contrário.
     */
    suspend fun indexDocument(document: Document): Boolean
    
    /**
     * Indexa uma lista de documentos.
     * 
     * @param documents Lista de documentos a serem indexados.
     * @return Lista de resultados de indexação, correspondendo à ordem dos documentos fornecidos.
     */
    suspend fun indexDocuments(documents: List<Document>): List<Boolean>
    
    /**
     * Indexa conteúdo a partir de um arquivo.
     * 
     * @param filePath Caminho para o arquivo a ser indexado.
     * @param metadata Metadados opcionais a serem associados.
     * @return true se a indexação foi bem-sucedida, false caso contrário.
     */
    suspend fun indexFromFile(filePath: String, metadata: Map<String, Any> = emptyMap()): Boolean
    
    /**
     * Indexa conteúdo a partir de texto.
     * 
     * @param text Texto a ser indexado.
     * @param id Identificador opcional (será gerado automaticamente se omitido).
     * @param metadata Metadados opcionais a serem associados.
     * @return true se a indexação foi bem-sucedida, false caso contrário.
     */
    suspend fun indexFromText(text: String, id: String? = null, metadata: Map<String, Any> = emptyMap()): Boolean
    
    /**
     * Realiza uma consulta RAG.
     * 
     * @param question A pergunta a ser respondida.
     * @return Resposta RAG contendo a resposta gerada e documentos relevantes.
     */
    suspend fun query(question: String): RAGResponse
    
    /**
     * Realiza uma consulta RAG com filtro por metadados.
     * 
     * @param question A pergunta a ser respondida.
     * @param filter Filtro de metadados para restringir a busca.
     * @return Resposta RAG contendo a resposta gerada e documentos relevantes.
     */
    suspend fun query(question: String, filter: Map<String, Any>?): RAGResponse
    
    /**
     * Realiza uma consulta RAG com opções personalizadas.
     * 
     * @param question A pergunta a ser respondida.
     * @param options Opções de consulta personalizadas.
     * @return Resposta RAG contendo a resposta gerada e documentos relevantes.
     */
    suspend fun query(question: String, options: QueryOptions): RAGResponse
    
    /**
     * Função interna para processar um documento com chunking antes da indexação.
     * 
     * @param document O documento a ser processado e indexado.
     * @return true se o processamento e indexação foram bem-sucedidos, false caso contrário.
     */
    suspend fun indexDocumentWithChunking(document: Document): Boolean
}
