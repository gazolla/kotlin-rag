package com.gazapps.rag.core.utils

import kotlinx.coroutines.delay
import kotlinx.coroutines.runBlocking
import org.junit.jupiter.api.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import kotlin.test.assertFailsWith

class AsyncBatchProcessorTest {

    @Test
    fun `processBatch should handle success and failure`() = runBlocking {
        val items = listOf(1, 2, 3, 4, 5)
        
        // Processor that succeeds for even numbers and fails for odd numbers
        val processor: suspend (Int) -> Int = { n ->
            if (n % 2 == 0) {
                n * 2
            } else {
                throw IllegalArgumentException("Odd number: $n")
            }
        }
        
        // Process with default config
        val results = AsyncBatchProcessor.processBatch(items, processor = processor)
        
        // Check results
        assertEquals(5, results.size)
        
        // Even numbers should succeed
        results.filter { it is AsyncBatchProcessor.BatchResult.Success }.forEach { result ->
            result as AsyncBatchProcessor.BatchResult.Success
            assertTrue(result.item % 2 == 0)
            assertEquals(result.item * 2, result.result)
        }
        
        // Odd numbers should fail
        results.filter { it is AsyncBatchProcessor.BatchResult.Failure }.forEach { result ->
            result as AsyncBatchProcessor.BatchResult.Failure
            assertTrue(result.item % 2 == 1)
            assertTrue(result.error is IllegalArgumentException)
        }
    }
    
    @Test
    fun `processInBatches should process items in batches`() = runBlocking {
        val items = (1..20).toList()
        
        // Process with batch size 5
        val results = AsyncBatchProcessor.processInBatches(
            items = items,
            batchSize = 5,
            processor = { batch ->
                // Double each number in the batch
                batch.map { it * 2 }
            }
        )
        
        // Check results
        assertEquals(20, results.size)
        items.forEachIndexed { index, item ->
            assertEquals(item * 2, results[index])
        }
    }
    
    @Test
    fun `executeWithCircuitBreaker should trip after threshold failures`() = runBlocking {
        var callCount = 0
        
        // Create an operation that always fails
        val failingOperation: suspend () -> String = {
            callCount++
            throw RuntimeException("Simulated failure")
        }
        
        // Configure circuit breaker with low threshold
        val config = AsyncBatchProcessor.CircuitBreakerConfig(
            failureThreshold = 2,
            resetTimeout = kotlin.time.Duration.milliseconds(100)
        )
        
        // First attempt - should just throw the exception
        assertFailsWith<RuntimeException> {
            AsyncBatchProcessor.executeWithCircuitBreaker(config, failingOperation)
        }
        
        // Second attempt - should throw and trip the circuit breaker
        assertFailsWith<RuntimeException> {
            AsyncBatchProcessor.executeWithCircuitBreaker(config, failingOperation)
        }
        
        // Third attempt - should throw CircuitBreakerOpenException
        assertFailsWith<AsyncBatchProcessor.CircuitBreakerOpenException> {
            AsyncBatchProcessor.executeWithCircuitBreaker(config, failingOperation)
        }
        
        // Verify that the operation was only called twice
        assertEquals(2, callCount)
        
        // Wait for the circuit breaker to reset
        delay(150)
        
        // After timeout, should try again (half-open state)
        assertFailsWith<RuntimeException> {
            AsyncBatchProcessor.executeWithCircuitBreaker(config, failingOperation)
        }
        
        // Should have called the operation again
        assertEquals(3, callCount)
    }
    
    @Test
    fun `batch processor should respect concurrency limits`() = runBlocking {
        val items = (1..10).toList()
        var maxConcurrent = 0
        var currentConcurrent = 0
        
        // Process with concurrency 3
        val config = AsyncBatchProcessor.BatchConfig(
            concurrency = 3
        )
        
        val results = AsyncBatchProcessor.processBatch(
            items = items,
            config = config,
            processor = { n ->
                synchronized(this) {
                    currentConcurrent++
                    if (currentConcurrent > maxConcurrent) {
                        maxConcurrent = currentConcurrent
                    }
                }
                
                // Simulate some work
                delay(50)
                
                synchronized(this) {
                    currentConcurrent--
                }
                
                n * 2
            }
        )
        
        // Check results
        assertEquals(10, results.size)
        results.forEach { result ->
            assertTrue(result is AsyncBatchProcessor.BatchResult.Success)
            result as AsyncBatchProcessor.BatchResult.Success
            assertEquals(result.item * 2, result.result)
        }
        
        // Check that concurrency limit was respected
        assertTrue(maxConcurrent <= 3)
    }
}
