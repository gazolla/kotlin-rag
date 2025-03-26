package com.gazapps.rag.core.utils

import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.delay
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.system.measureTimeMillis

class BatchProcessorTest {
    
    @Test
    fun `processBatches should process all items in batches`() = runBlocking {
        // Setup
        val items = (1..100).toList()
        
        // Act
        val results = BatchProcessor.processBatches(
            items = items,
            batchSize = 10,
            concurrency = 2,
            processor = { batch ->
                // Simple processor that just returns the input batch
                batch.map { it * 2 }
            }
        )
        
        // Assert
        assertEquals(100, results.size)
        assertEquals((1..100).map { it * 2 }, results)
    }
    
    @Test
    fun `processBatches should handle empty list`() = runBlocking {
        // Setup
        val items = emptyList<Int>()
        
        // Act
        val results = BatchProcessor.processBatches(
            items = items,
            processor = { batch ->
                batch.map { it * 2 }
            }
        )
        
        // Assert
        assertEquals(0, results.size)
    }
    
    @Test
    fun `processWithRateLimit should respect rate limit`() = runBlocking {
        // Setup
        val items = (1..10).toList()
        val requestsPerSecond = 5 // 5 requests per second = 200ms per request
        
        // Act
        val elapsedTime = measureTimeMillis {
            BatchProcessor.processWithRateLimit(
                items = items,
                requestsPerSecond = requestsPerSecond,
                concurrency = 1, // Sequential processing to test rate limiting
                processor = { item ->
                    // Simple processor that just returns the input item
                    item * 2
                }
            )
        }
        
        // Assert
        // For 10 items at 5 RPS, we expect at least 1800ms (9 delays of 200ms)
        // We allow some margin for test environment variations
        assertEquals(true, elapsedTime >= 1800, "Expected at least 1800ms, but got ${elapsedTime}ms")
    }
    
    @Test
    fun `processBatchesWithRateLimit should process batches with rate limit`() = runBlocking {
        // Setup
        val items = (1..40).toList()
        val batchSize = 10
        val requestsPerSecond = 5 // 5 batches per second = 200ms per batch
        
        // Act
        val results = BatchProcessor.processBatchesWithRateLimit(
            items = items,
            batchSize = batchSize,
            requestsPerSecond = requestsPerSecond,
            processor = { batch ->
                batch.map { it * 2 }
            }
        )
        
        // Assert
        assertEquals(40, results.size)
        assertEquals((1..40).map { it * 2 }, results)
    }
    
    @Test
    fun `processBatches should run batches in parallel`() = runBlocking {
        // Setup
        val items = (1..100).toList()
        val batchSize = 20
        val concurrency = 4
        val processingTime = 100L // ms per batch
        
        // Act
        val startTime = System.currentTimeMillis()
        
        BatchProcessor.processBatches(
            items = items,
            batchSize = batchSize,
            concurrency = concurrency,
            processor = { batch ->
                // Simulate processing time
                delay(processingTime)
                batch.map { it * 2 }
            }
        )
        
        val elapsedTime = System.currentTimeMillis() - startTime
        
        // Assert
        // With 5 batches (each taking 100ms) and concurrency of 4,
        // we expect around 200ms (2 rounds of processing)
        // Allow some margin for test environment variations
        val expectedMaxTime = 300L // 2 rounds + margin
        assertEquals(true, elapsedTime < expectedMaxTime, 
            "Expected less than ${expectedMaxTime}ms, but took ${elapsedTime}ms")
    }
}
