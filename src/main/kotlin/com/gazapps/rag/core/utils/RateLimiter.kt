package com.gazapps.rag.core.utils

import kotlinx.coroutines.delay
import java.util.concurrent.atomic.AtomicLong

/**
 * Utility class for rate limiting API calls
 *
 * @param requestsPerMinute Maximum number of requests allowed per minute
 */
class RateLimiter(requestsPerMinute: Int) {
    private val intervalMs: Long = (60.0 * 1000.0 / requestsPerMinute).toLong()
    private val lastRequestTime = AtomicLong(0)
    
    /**
     * Acquire permission to make a request, waiting if necessary to respect the rate limit
     */
    suspend fun acquire() {
        val currentTime = System.currentTimeMillis()
        val lastTime = lastRequestTime.get()
        val waitTime = lastTime + intervalMs - currentTime
        
        if (waitTime > 0) {
            delay(waitTime)
        }
        
        lastRequestTime.set(System.currentTimeMillis())
    }
}
