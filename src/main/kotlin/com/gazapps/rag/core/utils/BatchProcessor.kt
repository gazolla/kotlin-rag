package com.gazapps.rag.core.utils

import kotlinx.coroutines.*
import kotlinx.coroutines.sync.Semaphore
import org.slf4j.LoggerFactory
import java.time.Duration
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger

/**
 * Utility class for processing items in batches with coroutines
 */
object BatchProcessor {
    private val logger = LoggerFactory.getLogger(BatchProcessor::class.java)
    
    /**
     * Process a list of items in batches with specified parallelism
     * 
     * @param items The items to process
     * @param batchSize The size of each batch
     * @param concurrency Number of concurrent batch operations
     * @param dispatcher The coroutine dispatcher to use
     * @param processor Function to process each batch
     * @return List of results from the processor
     */
    suspend fun <T, R> processBatches(
        items: List<T>,
        batchSize: Int = 20,
        concurrency: Int = 4,
        dispatcher: CoroutineDispatcher = Dispatchers.Default,
        processor: suspend (List<T>) -> List<R>
    ): List<R> {
        if (items.isEmpty()) {
            return emptyList()
        }
        
        val startTime = System.currentTimeMillis()
        logger.debug("Starting batch processing of ${items.size} items with batchSize=$batchSize and concurrency=$concurrency")
        
        val batches = items.chunked(batchSize)
        val activeJobs = AtomicInteger(0)
        
        // Process in batches with concurrency control
        val results = coroutineScope {
            val deferreds = batches.map { batch ->
                async(dispatcher) {
                    activeJobs.incrementAndGet()
                    try {
                        processor(batch)
                    } finally {
                        activeJobs.decrementAndGet()
                    }
                }
            }
            
            deferreds.awaitAll().flatten()
        }
        
        val duration = Duration.ofMillis(System.currentTimeMillis() - startTime)
        logger.debug("Completed batch processing in ${duration.toMillis()}ms, processed ${items.size} items")
        
        return results
    }
    
    /**
     * Process items with rate limiting to avoid overloading external services
     * 
     * @param items The items to process
     * @param processor Function to process each item
     * @param requestsPerSecond Maximum requests per second
     * @param concurrency Number of concurrent operations
     * @return List of results from the processor
     */
    suspend fun <T, R> processWithRateLimit(
        items: List<T>,
        processor: suspend (T) -> R,
        requestsPerSecond: Int = 10,
        concurrency: Int = 4
    ): List<R> {
        if (items.isEmpty()) {
            return emptyList()
        }
        
        val startTime = System.currentTimeMillis()
        logger.debug("Starting rate-limited processing of ${items.size} items at $requestsPerSecond RPS")
        
        // Calculate delay between requests to maintain rate limit
        val delayMs = (1000.0 / requestsPerSecond).toLong()
        
        val results = mutableListOf<R>()
        
        withContext(Dispatchers.Default) {
            val semaphore = Semaphore(concurrency)
            
            val jobs = items.map { item ->
                launch {
                    semaphore.acquire()
                    try {
                        val result = processor(item)
                        synchronized(results) {
                            results.add(result)
                        }
                        delay(delayMs) // Add delay to maintain rate limit
                    } finally {
                        semaphore.release()
                    }
                }
            }
            
            jobs.joinAll()
        }
        
        val duration = Duration.ofMillis(System.currentTimeMillis() - startTime)
        logger.debug("Completed rate-limited processing in ${duration.toMillis()}ms, processed ${items.size} items")
        
        return results
    }
    
    /**
     * Process items with both batching and rate limiting
     * 
     * @param items The items to process
     * @param batchSize The size of each batch
     * @param requestsPerSecond Maximum requests per second
     * @param processor Function to process each batch
     * @return List of results from the processor
     */
    suspend fun <T, R> processBatchesWithRateLimit(
        items: List<T>,
        batchSize: Int = 20,
        requestsPerSecond: Int = 5,
        processor: suspend (List<T>) -> List<R>
    ): List<R> {
        if (items.isEmpty()) {
            return emptyList()
        }
        
        val startTime = System.currentTimeMillis()
        logger.debug("Starting batch processing with rate limiting, ${items.size} items, batchSize=$batchSize, RPS=$requestsPerSecond")
        
        val batches = items.chunked(batchSize)
        val delayMs = (1000.0 / requestsPerSecond).toLong()
        
        val results = mutableListOf<R>()
        
        for (batch in batches) {
            val batchResults = processor(batch)
            results.addAll(batchResults)
            delay(delayMs)
        }
        
        val duration = Duration.ofMillis(System.currentTimeMillis() - startTime)
        logger.debug("Completed batch processing with rate limiting in ${duration.toMillis()}ms, processed ${items.size} items")
        
        return results
    }
}
