package com.gazapps.rag.core.config

import com.gazapps.rag.core.document.ChunkingStrategy
import com.gazapps.rag.core.document.PreprocessingConfig

/**
 * Configuração específica para o processo de indexação de documentos.
 */
data class IndexingConfig(
    /**
     * Tamanho máximo dos chunks durante o processamento de documentos.
     */
    var chunkSize: Int = 500,
    
    /**
     * Quantidade de sobreposição entre chunks em tokens.
     */
    var chunkOverlap: Int = 50,
    
    /**
     * Estratégia utilizada para dividir documentos em chunks.
     */
    var chunkingStrategy: ChunkingStrategy = ChunkingStrategy.PARAGRAPH,
    
    /**
     * Se deve realizar pré-processamento de texto antes da indexação.
     */
    var preprocessText: Boolean = true,
    
    /**
     * Configuração para o pré-processamento de texto.
     */
    var textPreprocessingConfig: PreprocessingConfig = PreprocessingConfig(),
    
    /**
     * Se deve processar documentos de forma assíncrona.
     */
    var asyncProcessing: Boolean = false,
    
    /**
     * Tamanho do lote para processamento assíncrono.
     */
    var asyncBatchSize: Int = 10,
    
    /**
     * Número de operações concorrentes durante o processamento assíncrono.
     */
    var asyncConcurrency: Int = 4,
    
    /**
     * Timeout para operações assíncronas em milissegundos.
     */
    var asyncTimeout: Long = 60000 // 60 segundos
)
