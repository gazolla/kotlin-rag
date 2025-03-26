package com.gazapps.rag.core.utils

import kotlinx.coroutines.*
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.flow.*
import kotlinx.coroutines.sync.Semaphore
import kotlinx.coroutines.sync.withPermit
import org.slf4j.LoggerFactory
import java.time.Duration
import java.util.concurrent.atomic.AtomicInteger
import kotlin.math.min

/**
 * Advanced batch processor with more sophisticated features for managing
 * parallel processing, error handling, and flow control
 */
object AdvancedBatchProcessor {
    private val logger = LoggerFactory.getLogger(AdvancedBatchProcessor::class.java)

    /**
     * Process items in a flow with controlled concurrency and backpressure
     *
     * @param items The items to process
     * @param concurrency Maximum number of concurrent operations
     * @param bufferSize Size of the flow buffer for backpressure control
     * @param processor Function to process each item
     * @return Flow of processed results
     */
    fun <T, R> processAsFlow(
        items: List<T>,
        concurrency: Int = 4,
        bufferSize: Int = Channel.BUFFERED,
        processor: suspend (T) -> R
    ): Flow<Result<R>> = flow<Result<R>> {
        val semaphore = Semaphore(concurrency)
        
        coroutineScope {
            items.forEach { item ->
                semaphore.acquire()
                launch {
                    try {
                        val result = processor(item)
                        emit(Result.success(result))
                    } catch (e: Exception) {
                        emit(Result.failure(e))
                    } finally {
                        semaphore.release()
                    }
                }
            }
        }
    }.buffer(bufferSize)
    
    /**
     * Process items with adaptive concurrency that adjusts based on success/failure rates
     *
     * @param items The items to process
     * @param initialConcurrency Starting concurrency level
     * @param minConcurrency Minimum concurrency level
     * @param maxConcurrency Maximum concurrency level
     * @param adaptationFactor How quickly to adapt (higher = faster)
     * @param processor Function to process each item
     * @return List of successful results
     */
    suspend fun <T, R> processWithAdaptiveConcurrency(
        items: List<T>,
        initialConcurrency: Int = 4,
        minConcurrency: Int = 1,
        maxConcurrency: Int = 16,
        adaptationFactor: Float = 0.1f,
        processor: suspend (T) -> R
    ): List<R> {
        if (items.isEmpty()) return emptyList()
        
        val results = mutableListOf<R>()
        var currentConcurrency = initialConcurrency
        val successCount = AtomicInteger(0)
        val failureCount = AtomicInteger(0)
        
        // Process in batches to allow for concurrency adjustments
        val batchSize = 10
        val batches = items.chunked(batchSize)
        
        for (batch in batches) {
            // Use a semaphore to limit concurrency
            val semaphore = Semaphore(currentConcurrency)
            
            coroutineScope {
                batch.forEach { item ->
                    launch {
                        semaphore.withPermit {
                            try {
                                val result = processor(item)
                                synchronized(results) {
                                    results.add(result)
                                }
                                successCount.incrementAndGet()
                            } catch (e: Exception) {
                                logger.warn("Processing error: ${e.message}")
                                failureCount.incrementAndGet()
                            }
                        }
                    }
                }
            }
            
            // Adjust concurrency based on success/failure ratio
            val totalProcessed = successCount.get() + failureCount.get()
            if (totalProcessed > 0) {
                val successRate = successCount.get().toFloat() / totalProcessed
                
                // Increase concurrency if success rate is high, decrease if low
                val adjustment = ((successRate - 0.5f) * 2 * adaptationFactor * currentConcurrency).toInt()
                
                // Apply the adjustment, staying within bounds
                currentConcurrency = (currentConcurrency + adjustment)
                    .coerceIn(minConcurrency, maxConcurrency)
                
                logger.debug("Adjusted concurrency to $currentConcurrency (success rate: ${successRate * 100}%)")
            }
        }
        
        return results
    }
    
    /**
     * Process items with backoff and retry for failures
     *
     * @param items The items to process
     * @param maxRetries Maximum number of retries per item
     * @param initialBackoffMs Initial backoff time in milliseconds
     * @param maxBackoffMs Maximum backoff time in milliseconds
     * @param processor Function to process each item
     * @return List of results and any remaining failures
     */
    suspend fun <T, R> processWithRetry(
        items: List<T>,
        maxRetries: Int = 3,
        initialBackoffMs: Long = 100,
        maxBackoffMs: Long = 5000,
        processor: suspend (T) -> R
    ): Pair<List<R>, List<Pair<T, Exception>>> {
        if (items.isEmpty()) return Pair(emptyList(), emptyList())
        
        val results = mutableListOf<R>()
        val failures = mutableListOf<Pair<T, Exception>>()
        
        // Initial attempt
        val itemsWithRetries = items.associateWith { 0 }.toMutableMap()
        
        // Continue until all items are processed or max retries reached
        while (itemsWithRetries.isNotEmpty()) {
            val batch = itemsWithRetries.keys.toList()
            val batchResults = mutableMapOf<T, Result<R>>()
            
            // Process current batch
            coroutineScope {
                batch.forEach { item ->
                    launch {
                        try {
                            val result = processor(item)
                            synchronized(batchResults) {
                                batchResults[item] = Result.success(result)
                            }
                        } catch (e: Exception) {
                            synchronized(batchResults) {
                                batchResults[item] = Result.failure(e)
                            }
                        }
                    }
                }
            }
            
            // Process results
            for (item in batch) {
                val result = batchResults[item]
                if (result == null) {
                    // This shouldn't happen, but handle it anyway
                    val retries = itemsWithRetries[item] ?: 0
                    if (retries >= maxRetries) {
                        failures.add(item to Exception("Unknown failure"))
                        itemsWithRetries.remove(item)
                    } else {
                        itemsWithRetries[item] = retries + 1
                    }
                    continue
                }
                
                if (result.isSuccess) {
                    // Success, add to results
                    results.add(result.getOrThrow())
                    itemsWithRetries.remove(item)
                } else {
                    // Failure, increment retry count
                    val retries = itemsWithRetries[item] ?: 0
                    val exception = result.exceptionOrNull() as? Exception ?: Exception("Unknown failure")
                    
                    if (retries >= maxRetries) {
                        failures.add(item to exception)
                        itemsWithRetries.remove(item)
                    } else {
                        itemsWithRetries[item] = retries + 1
                    }
                }
            }
            
            // If there are items to retry, apply backoff
            if (itemsWithRetries.isNotEmpty()) {
                // Calculate backoff based on maximum retry count of remaining items
                val maxCurrentRetry = itemsWithRetries.values.maxOrNull() ?: 0
                val backoffMs = min(initialBackoffMs * (1 shl maxCurrentRetry), maxBackoffMs)
                logger.debug("Backing off for ${backoffMs}ms before retrying ${itemsWithRetries.size} items")
                delay(backoffMs)
            }
        }
        
        return Pair(results, failures)
    }
    
    /**
     * Process items with prioritization
     *
     * @param items The items to process
     * @param prioritySelector Function that assigns a priority to each item (higher = more important)
     * @param processor Function to process each item
     * @return List of results in their original order
     */
    suspend fun <T, R> processWithPriority(
        items: List<T>,
        prioritySelector: (T) -> Int,
        processor: suspend (T) -> R
    ): List<R> {
        if (items.isEmpty()) return emptyList()
        
        // Create entries with original indices and priorities
        data class Entry(val item: T, val index: Int, val priority: Int)
        
        val entries = items.mapIndexed { index, item ->
            Entry(item, index, prioritySelector(item))
        }
        
        // Sort by priority (descending)
        val sortedEntries = entries.sortedByDescending { it.priority }
        
        // Process in priority order
        val processedEntries = coroutineScope {
            sortedEntries.map { entry ->
                async {
                    val result = processor(entry.item)
                    entry.index to result
                }
            }.awaitAll()
        }
        
        // Restore original order
        val resultArray = arrayOfNulls<Any>(items.size)
        processedEntries.forEach { (index, result) ->
            @Suppress("UNCHECKED_CAST")
            resultArray[index] = result as Any
        }
        
        @Suppress("UNCHECKED_CAST")
        return resultArray.map { it as R }
    }
    
    /**
     * Process items with circuit breaker pattern to prevent cascading failures
     *
     * @param items The items to process
     * @param failureThreshold Number of failures that will trip the circuit breaker
     * @param resetTimeoutMs Time to wait before trying again after circuit breaker trips
     * @param processor Function to process each item
     * @return Results for successfully processed items
     */
    suspend fun <T, R> processWithCircuitBreaker(
        items: List<T>,
        failureThreshold: Int = 5,
        resetTimeoutMs: Long = 5000,
        processor: suspend (T) -> R
    ): List<R> {
        if (items.isEmpty()) return emptyList()
        
        val results = mutableListOf<R>()
        val circuitBreaker = CircuitBreaker(failureThreshold, resetTimeoutMs)
        
        for (item in items) {
            try {
                val result = circuitBreaker.execute { processor(item) }
                results.add(result)
            } catch (e: CircuitBreakerOpenException) {
                logger.warn("Circuit breaker open, skipping remaining items")
                break
            } catch (e: Exception) {
                logger.warn("Processing error: ${e.message}")
                // Continue with next item
            }
        }
        
        return results
    }
    
    /**
     * Simple circuit breaker implementation
     */
    private class CircuitBreaker(
        private val failureThreshold: Int,
        private val resetTimeoutMs: Long
    ) {
        private enum class State { CLOSED, OPEN, HALF_OPEN }
        
        private var state = State.CLOSED
        private var failureCount = 0
        private var lastFailureTime = 0L
        
        suspend fun <T> execute(block: suspend () -> T): T {
            when (state) {
                State.OPEN -> {
                    // Check if it's time to try again
                    if (System.currentTimeMillis() - lastFailureTime > resetTimeoutMs) {
                        state = State.HALF_OPEN
                    } else {
                        throw CircuitBreakerOpenException("Circuit breaker is open")
                    }
                }
                State.HALF_OPEN, State.CLOSED -> {
                    // Proceed with execution
                }
            }
            
            try {
                val result = block()
                
                // Success, reset failure count
                state = State.CLOSED
                failureCount = 0
                
                return result
            } catch (e: Exception) {
                handleFailure()
                throw e
            }
        }
        
        private fun handleFailure() {
            lastFailureTime = System.currentTimeMillis()
            
            when (state) {
                State.CLOSED -> {
                    failureCount++
                    if (failureCount >= failureThreshold) {
                        state = State.OPEN
                    }
                }
                State.HALF_OPEN -> {
                    state = State.OPEN
                }
                State.OPEN -> {
                    // Already open, just update the time
                }
            }
        }
    }
    
    class CircuitBreakerOpenException(message: String) : Exception(message)
}
