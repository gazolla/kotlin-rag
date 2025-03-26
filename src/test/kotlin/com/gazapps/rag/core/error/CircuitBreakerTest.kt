package com.gazapps.rag.core.error

import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.delay
import java.io.IOException

class CircuitBreakerTest {
    
    @Test
    fun `circuit starts in closed state`() {
        val circuitBreaker = CircuitBreaker()
        assertEquals(CircuitBreakerState.CLOSED, circuitBreaker.getState())
        assertTrue(circuitBreaker.allowRequest())
        assertEquals(0, circuitBreaker.getFailureCount())
    }
    
    @Test
    fun `circuit opens after reaching failure threshold`() {
        val circuitBreaker = CircuitBreaker(failureThreshold = 3)
        
        // Record failures
        for (i in 1..3) {
            circuitBreaker.recordFailure(IOException("Failure $i"))
        }
        
        assertEquals(CircuitBreakerState.OPEN, circuitBreaker.getState())
        assertFalse(circuitBreaker.allowRequest())
        assertEquals(3, circuitBreaker.getFailureCount())
    }
    
    @Test
    fun `circuit allows limited requests in half-open state`() = runBlocking {
        val circuitBreaker = CircuitBreaker(
            failureThreshold = 2,
            resetTimeoutMs = 100, // Short timeout for testing
            halfOpenMaxRequests = 2
        )
        
        // Open the circuit
        circuitBreaker.recordFailure(IOException("Failure 1"))
        circuitBreaker.recordFailure(IOException("Failure 2"))
        
        assertEquals(CircuitBreakerState.OPEN, circuitBreaker.getState())
        assertFalse(circuitBreaker.allowRequest())
        
        // Wait for reset timeout
        delay(150)
        
        // Should be half-open now
        assertEquals(CircuitBreakerState.HALF_OPEN, circuitBreaker.getState())
        
        // Should allow halfOpenMaxRequests requests
        assertTrue(circuitBreaker.allowRequest())
        assertTrue(circuitBreaker.allowRequest())
        
        // Should reject additional requests
        assertFalse(circuitBreaker.allowRequest())
    }
    
    @Test
    fun `circuit closes after successful request in half-open state`() = runBlocking {
        val circuitBreaker = CircuitBreaker(
            failureThreshold = 2,
            resetTimeoutMs = 100,
            halfOpenMaxRequests = 1
        )
        
        // Open the circuit
        circuitBreaker.recordFailure(IOException("Failure 1"))
        circuitBreaker.recordFailure(IOException("Failure 2"))
        
        assertEquals(CircuitBreakerState.OPEN, circuitBreaker.getState())
        
        // Wait for reset timeout
        delay(150)
        
        // Should be half-open now
        assertEquals(CircuitBreakerState.HALF_OPEN, circuitBreaker.getState())
        assertTrue(circuitBreaker.allowRequest())
        
        // Record a success
        circuitBreaker.recordSuccess()
        
        // Circuit should be closed
        assertEquals(CircuitBreakerState.CLOSED, circuitBreaker.getState())
        assertEquals(0, circuitBreaker.getFailureCount())
        assertTrue(circuitBreaker.allowRequest())
    }
    
    @Test
    fun `circuit reopens after failure in half-open state`() = runBlocking {
        val circuitBreaker = CircuitBreaker(
            failureThreshold = 2,
            resetTimeoutMs = 100,
            halfOpenMaxRequests = 1
        )
        
        // Open the circuit
        circuitBreaker.recordFailure(IOException("Failure 1"))
        circuitBreaker.recordFailure(IOException("Failure 2"))
        
        // Wait for reset timeout
        delay(150)
        
        // Should be half-open now
        assertEquals(CircuitBreakerState.HALF_OPEN, circuitBreaker.getState())
        assertTrue(circuitBreaker.allowRequest())
        
        // Record another failure
        circuitBreaker.recordFailure(IOException("Failure in half-open"))
        
        // Circuit should be open again
        assertEquals(CircuitBreakerState.OPEN, circuitBreaker.getState())
        assertFalse(circuitBreaker.allowRequest())
    }
    
    @Test
    fun `circuit breaker registry returns same instance for same name`() {
        val cb1 = CircuitBreakerRegistry.getCircuitBreaker("test-service")
        val cb2 = CircuitBreakerRegistry.getCircuitBreaker("test-service")
        
        assertSame(cb1, cb2)
    }
    
    @Test
    fun `circuit breaker registry returns different instances for different names`() {
        val cb1 = CircuitBreakerRegistry.getCircuitBreaker("service-1")
        val cb2 = CircuitBreakerRegistry.getCircuitBreaker("service-2")
        
        assertNotSame(cb1, cb2)
    }
}
