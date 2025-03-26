package com.gazapps.rag.core.error

import kotlinx.coroutines.delay
import kotlin.math.pow
import kotlin.random.Random

/**
 * Utility for consistent error handling across the RAG library.
 * Provides retry, circuit breaking, and fallback mechanisms.
 */
class ErrorHandler(
    private val componentName: String = "ErrorHandler",
    private val logger: RAGLogger = RAGLoggerFactory.getLogger()
) {
    /**
     * Configuration for retry operations
     */
    data class RetryConfig(
        val maxRetries: Int = 3,
        val initialDelayMs: Long = 100,
        val maxDelayMs: Long = 10000,
        val factor: Double = 2.0,
        val jitter: Double = 0.1
    )
    
    /**
     * Execute with fallback: if the primary operation fails, execute the fallback
     */
    suspend fun <T> withFallback(primary: suspend () -> T, fallback: suspend (Exception) -> T): T {
        return try {
            primary()
        } catch (e: Exception) {
            logger.warn("Primary operation failed, executing fallback: ${e.message}", componentName)
            fallback(e)
        }
    }
    
    /**
     * Execute with retry: retry the operation with exponential backoff
     */
    suspend fun <T> withRetry(
        config: RetryConfig = RetryConfig(),
        block: suspend () -> T
    ): T {
        var currentDelay = config.initialDelayMs
        var attempts = 0
        
        while (true) {
            try {
                return block()
            } catch (e: Exception) {
                attempts++
                
                if (attempts >= config.maxRetries) {
                    logger.error("Operation failed after ${attempts} attempts: ${e.message}", componentName)
                    throw e
                }
                
                logger.warn("Retry ${attempts}/${config.maxRetries} after error: ${e.message}", componentName)
                
                // Add jitter to avoid thundering herd
                val jitterFactor = 1.0 + Random.nextDouble(-config.jitter, config.jitter)
                val delay = (currentDelay * jitterFactor).toLong().coerceAtMost(config.maxDelayMs)
                
                delay(delay)
                
                // Exponential backoff
                currentDelay = (currentDelay * config.factor).toLong().coerceAtMost(config.maxDelayMs)
            }
        }
    }
    
    /**
     * Execute with retry for specific exceptions: only retry for the given exception types
     */
    suspend fun <T> withRetry(
        retryFor: List<Class<out Exception>>,
        config: RetryConfig = RetryConfig(),
        block: suspend () -> T
    ): T {
        var currentDelay = config.initialDelayMs
        var attempts = 0
        
        while (true) {
            try {
                return block()
            } catch (e: Exception) {
                val shouldRetry = retryFor.any { it.isInstance(e) }
                
                if (!shouldRetry) {
                    throw e
                }
                
                attempts++
                
                if (attempts >= config.maxRetries) {
                    logger.error("Operation failed after ${attempts} attempts: ${e.message}", componentName)
                    throw e
                }
                
                logger.warn("Retry ${attempts}/${config.maxRetries} after error: ${e.message}", componentName)
                
                // Add jitter to avoid thundering herd
                val jitterFactor = 1.0 + Random.nextDouble(-config.jitter, config.jitter)
                val delay = (currentDelay * jitterFactor).toLong().coerceAtMost(config.maxDelayMs)
                
                delay(delay)
                
                // Exponential backoff
                currentDelay = (currentDelay * config.factor).toLong().coerceAtMost(config.maxDelayMs)
            }
        }
    }
    
    /**
     * Execute with circuit breaker: stop trying if too many failures occur
     */
    suspend fun <T> withCircuitBreaker(
        circuitBreaker: CircuitBreaker,
        block: suspend () -> T
    ): T {
        return circuitBreaker.execute(block)
    }
}
