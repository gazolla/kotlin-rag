package com.gazapps.rag.core.error

import kotlinx.coroutines.runBlocking
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.assertThrows
import java.io.IOException
import java.net.ConnectException

class ErrorHandlerTest {
    
    private val errorHandler = ErrorHandler(ConsoleLogger(LogLevel.NONE))
    
    @Test
    fun `withRetry should retry the operation and succeed eventually`() = runBlocking {
        // Counter to track number of attempts
        var attempts = 0
        
        // Function that fails twice, then succeeds
        val result = errorHandler.withRetry(
            operation = "test-operation",
            config = ErrorHandler.RetryConfig(maxRetries = 3)
        ) {
            attempts++
            if (attempts < 3) {
                throw IOException("Simulated failure")
            }
            "success"
        }
        
        assertEquals("success", result)
        assertEquals(3, attempts)
    }
    
    @Test
    fun `withRetry should fail after max retries`() = runBlocking {
        var attempts = 0
        
        val exception = assertThrows<IOException> {
            errorHandler.withRetry(
                operation = "test-operation",
                config = ErrorHandler.RetryConfig(maxRetries = 2)
            ) {
                attempts++
                throw IOException("Simulated failure")
            }
        }
        
        assertEquals("Simulated failure", exception.message)
        assertEquals(3, attempts) // Initial attempt + 2 retries = 3 attempts
    }
    
    @Test
    fun `withRetry should not retry if shouldRetry returns false`() = runBlocking {
        var attempts = 0
        
        val exception = assertThrows<IOException> {
            errorHandler.withRetry(
                operation = "test-operation",
                config = ErrorHandler.RetryConfig(maxRetries = 2),
                shouldRetry = { e -> e !is IOException } // Don't retry IOExceptions
            ) {
                attempts++
                throw IOException("Simulated failure")
            }
        }
        
        assertEquals("Simulated failure", exception.message)
        assertEquals(1, attempts) // Only one attempt, no retries
    }
    
    @Test
    fun `withFallback should use fallback when primary fails`() = runBlocking {
        val result = errorHandler.withFallback(
            operation = "test-operation",
            primary = { throw IOException("Primary failed") },
            fallback = { ex -> 
                assertEquals("Primary failed", ex.message)
                "fallback-result"
            }
        )
        
        assertEquals("fallback-result", result)
    }
    
    @Test
    fun `withFallback should use primary when it succeeds`() = runBlocking {
        val result = errorHandler.withFallback(
            operation = "test-operation",
            primary = { "primary-result" },
            fallback = { ex -> "fallback-result" }
        )
        
        assertEquals("primary-result", result)
    }
    
    @Test
    fun `withFallback should propagate exception if both primary and fallback fail`() = runBlocking {
        val exception = assertThrows<IOException> {
            errorHandler.withFallback(
                operation = "test-operation",
                primary = { throw ConnectException("Primary failed") },
                fallback = { ex -> throw IOException("Fallback failed") }
            )
        }
        
        assertEquals("Fallback failed", exception.message)
    }
    
    @Test
    fun `withCircuitBreaker should allow requests when circuit is closed`() = runBlocking {
        val circuitBreaker = CircuitBreaker(failureThreshold = 3)
        
        val result = errorHandler.withCircuitBreaker(
            operation = "test-operation",
            circuitBreaker = circuitBreaker
        ) {
            "success"
        }
        
        assertEquals("success", result)
        assertEquals(CircuitBreakerState.CLOSED, circuitBreaker.getState())
    }
    
    @Test
    fun `withCircuitBreaker should record failure but still be closed if under threshold`() = runBlocking {
        val circuitBreaker = CircuitBreaker(failureThreshold = 3)
        
        // First failure
        val exception = assertThrows<IOException> {
            errorHandler.withCircuitBreaker(
                operation = "test-operation",
                circuitBreaker = circuitBreaker
            ) {
                throw IOException("Simulated failure")
            }
        }
        
        assertEquals("Simulated failure", exception.message)
        assertEquals(CircuitBreakerState.CLOSED, circuitBreaker.getState())
        assertEquals(1, circuitBreaker.getFailureCount())
    }
    
    @Test
    fun `withCircuitBreaker should open circuit after threshold failures`() = runBlocking {
        val circuitBreaker = CircuitBreaker(failureThreshold = 2)
        
        // First failure
        assertThrows<IOException> {
            errorHandler.withCircuitBreaker(
                operation = "test-operation",
                circuitBreaker = circuitBreaker
            ) {
                throw IOException("Failure 1")
            }
        }
        
        // Second failure - should open the circuit
        assertThrows<IOException> {
            errorHandler.withCircuitBreaker(
                operation = "test-operation",
                circuitBreaker = circuitBreaker
            ) {
                throw IOException("Failure 2")
            }
        }
        
        assertEquals(CircuitBreakerState.OPEN, circuitBreaker.getState())
        assertEquals(2, circuitBreaker.getFailureCount())
        
        // Next attempt should fail with ServiceUnavailableException
        val exception = assertThrows<RAGException.ServiceUnavailableException> {
            errorHandler.withCircuitBreaker(
                operation = "test-operation",
                circuitBreaker = circuitBreaker
            ) {
                "This should not be executed"
            }
        }
        
        assertTrue(exception.message!!.contains("circuit breaker is open"))
    }
    
    @Test
    fun `withCircuitBreaker should reset after success`() = runBlocking {
        val circuitBreaker = CircuitBreaker(failureThreshold = 3)
        
        // Two failures
        for (i in 1..2) {
            assertThrows<IOException> {
                errorHandler.withCircuitBreaker(
                    operation = "test-operation",
                    circuitBreaker = circuitBreaker
                ) {
                    throw IOException("Failure $i")
                }
            }
        }
        
        // Circuit still closed
        assertEquals(CircuitBreakerState.CLOSED, circuitBreaker.getState())
        assertEquals(2, circuitBreaker.getFailureCount())
        
        // Success should reset failure count
        val result = errorHandler.withCircuitBreaker(
            operation = "test-operation",
            circuitBreaker = circuitBreaker
        ) {
            "success"
        }
        
        assertEquals("success", result)
        assertEquals(CircuitBreakerState.CLOSED, circuitBreaker.getState())
        assertEquals(0, circuitBreaker.getFailureCount())
    }
}
