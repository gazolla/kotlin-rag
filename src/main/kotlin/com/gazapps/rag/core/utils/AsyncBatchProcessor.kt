package com.gazapps.rag.core.utils

import kotlinx.coroutines.*
import kotlin.time.Duration

/**
 * Processador assíncrono para operações em lote.
 */
object AsyncBatchProcessor {
    /**
     * Configuração para processamento em lote.
     */
    data class BatchConfig(
        /**
         * Número máximo de operações concorrentes.
         */
        val concurrency: Int = 4,
        
        /**
         * Tempo limite para cada operação.
         */
        val timeout: Duration = Duration.parse("30s"),
        
        /**
         * Número máximo de tentativas em caso de falha.
         */
        val maxRetries: Int = 3,
        
        /**
         * Tempo de espera entre tentativas em ms.
         */
        val retryDelayMs: Long = 1000
    )
    
    /**
     * Processa uma lista de itens em lotes.
     *
     * @param items Lista de itens a serem processados.
     * @param batchSize Tamanho de cada lote.
     * @param config Configuração de processamento.
     * @param processor Função de processamento para cada lote.
     * @return Lista dos resultados combinados.
     */
    suspend fun <T, R> processInBatches(
        items: List<T>,
        batchSize: Int,
        config: BatchConfig = BatchConfig(),
        processor: suspend (List<T>) -> List<R>
    ): List<R> {
        if (items.isEmpty()) return emptyList()
        
        val results = mutableListOf<R>()
        val batches = items.chunked(batchSize)
        
        coroutineScope {
            val semaphore = kotlinx.coroutines.sync.Semaphore(config.concurrency)
            
            val deferreds = batches.map { batch ->
                async {
                    semaphore.acquire()
                    try {
                        withTimeoutOrNull(config.timeout.inWholeMilliseconds) {
                            processBatchWithRetry(batch, config, processor)
                        } ?: emptyList() // Em caso de timeout, retornar lista vazia
                    } finally {
                        semaphore.release()
                    }
                }
            }
            
            deferreds.forEach { deferred ->
                results.addAll(deferred.await())
            }
        }
        
        return results
    }
    
    /**
     * Processa um lote com suporte a retry.
     */
    private suspend fun <T, R> processBatchWithRetry(
        batch: List<T>,
        config: BatchConfig,
        processor: suspend (List<T>) -> List<R>
    ): List<R> {
        var attempts = 0
        var lastException: Exception? = null
        
        while (attempts < config.maxRetries) {
            try {
                return processor(batch)
            } catch (e: Exception) {
                lastException = e
                attempts++
                
                if (attempts < config.maxRetries) {
                    delay(config.retryDelayMs * attempts) // Backoff exponencial
                }
            }
        }
        
        // Se chegamos aqui, todas as tentativas falharam
        throw lastException ?: IllegalStateException("Failed to process batch after ${config.maxRetries} attempts")
    }
}
