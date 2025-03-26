package com.gazapps.rag.core.error

import java.time.Instant
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicReference

/**
 * Circuit breaker states
 */
enum class CircuitBreakerState {
    CLOSED,     // Normal operation, all calls pass through
    OPEN,       // Circuit is open, all calls fail fast
    HALF_OPEN   // Testing state, limited calls pass through to test recovery
}

/**
 * Implementation of the Circuit Breaker pattern to prevent repeated calls to failing services.
 * 
 * @property failureThreshold Number of failures before circuit opens
 * @property resetTimeout Time in milliseconds before attempting to half-open the circuit
 * @property successThreshold Number of consecutive successes required to close circuit from half-open
 */
class CircuitBreaker(
    private val name: String,
    private val failureThreshold: Int = 5,
    private val resetTimeout: Long = 60000, // 1 minute
    private val successThreshold: Int = 2,
    private val logger: RAGLogger = RAGLoggerFactory.getLogger()
) {
    private val state = AtomicReference(CircuitBreakerState.CLOSED)
    private val failureCount = AtomicInteger(0)
    private val successCount = AtomicInteger(0)
    private val lastStateChange = AtomicReference(Instant.now())
    
    /**
     * Execute an operation with circuit breaker protection.
     * 
     * @param block Operation to execute
     * @return Result of the operation
     * @throws CircuitBreakerOpenException if circuit is open and operation is rejected
     */
    suspend fun <T> execute(block: suspend () -> T): T {
        when (state.get()) {
            CircuitBreakerState.OPEN -> {
                if (shouldAttemptReset()) {
                    return executeInHalfOpenState(block)
                } else {
                    logger.warn("Circuit $name is OPEN, fast failing", "CircuitBreaker")
                    throw CircuitBreakerOpenException("Circuit $name is open")
                }
            }
            CircuitBreakerState.HALF_OPEN -> {
                return executeInHalfOpenState(block)
            }
            CircuitBreakerState.CLOSED -> {
                return try {
                    block()
                } catch (e: Exception) {
                    handleFailure(e)
                    throw e
                }
            }
        }
    }
    
    private suspend fun <T> executeInHalfOpenState(block: suspend () -> T): T {
        logger.info("Circuit $name is HALF-OPEN, testing connection", "CircuitBreaker")
        return try {
            val result = block()
            handleSuccess()
            result
        } catch (e: Exception) {
            moveToOpenState()
            throw e
        }
    }
    
    private fun handleFailure(exception: Exception) {
        val count = failureCount.incrementAndGet()
        logger.warn("Operation failed on circuit $name, failure count: $count", "CircuitBreaker", exception)
        
        if (count >= failureThreshold) {
            moveToOpenState()
        }
    }
    
    private fun handleSuccess() {
        if (state.get() == CircuitBreakerState.HALF_OPEN) {
            val count = successCount.incrementAndGet()
            logger.info("Success on circuit $name in half-open state, success count: $count", "CircuitBreaker")
            
            if (count >= successThreshold) {
                moveToClosedState()
            }
        } else {
            // Reset counter when in closed state
            failureCount.set(0)
        }
    }
    
    private fun moveToOpenState() {
        if (state.compareAndSet(CircuitBreakerState.CLOSED, CircuitBreakerState.OPEN) ||
            state.compareAndSet(CircuitBreakerState.HALF_OPEN, CircuitBreakerState.OPEN)) {
            logger.warn("Circuit $name moved to OPEN state", "CircuitBreaker")
            lastStateChange.set(Instant.now())
            failureCount.set(0)
            successCount.set(0)
        }
    }
    
    private fun moveToClosedState() {
        if (state.compareAndSet(CircuitBreakerState.HALF_OPEN, CircuitBreakerState.CLOSED)) {
            logger.info("Circuit $name moved to CLOSED state", "CircuitBreaker")
            lastStateChange.set(Instant.now())
            failureCount.set(0)
            successCount.set(0)
        }
    }
    
    private fun moveToHalfOpenState() {
        if (state.compareAndSet(CircuitBreakerState.OPEN, CircuitBreakerState.HALF_OPEN)) {
            logger.info("Circuit $name moved to HALF-OPEN state", "CircuitBreaker")
            lastStateChange.set(Instant.now())
            successCount.set(0)
        }
    }
    
    private fun shouldAttemptReset(): Boolean {
        val now = Instant.now()
        val elapsed = now.toEpochMilli() - lastStateChange.get().toEpochMilli()
        
        if (elapsed >= resetTimeout) {
            moveToHalfOpenState()
            return true
        }
        
        return false
    }
    
    /**
     * Get the current state of the circuit breaker
     */
    fun getState(): CircuitBreakerState = state.get()
    
    /**
     * Exception thrown when the circuit is open
     */
    class CircuitBreakerOpenException(message: String) : Exception(message)
}

/**
 * Registry to manage multiple circuit breakers
 */
object CircuitBreakerRegistry {
    private val breakers = mutableMapOf<String, CircuitBreaker>()
    
    /**
     * Get a circuit breaker by name, creating it if it doesn't exist
     */
    fun getCircuitBreaker(
        name: String,
        failureThreshold: Int = 5,
        resetTimeout: Long = 60000,
        successThreshold: Int = 2
    ): CircuitBreaker {
        return breakers.getOrPut(name) {
            CircuitBreaker(name, failureThreshold, resetTimeout, successThreshold)
        }
    }
    
    /**
     * Clear all circuit breakers
     */
    fun clear() {
        breakers.clear()
    }
}
