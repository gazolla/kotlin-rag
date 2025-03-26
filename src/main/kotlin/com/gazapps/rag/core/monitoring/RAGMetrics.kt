package com.gazapps.rag.core.monitoring

import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicLong
import java.time.Instant
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock

/**
 * System for collecting and reporting metrics about RAG operations
 */
class RAGMetrics {
    private val counters = ConcurrentHashMap<String, AtomicLong>()
    private val timers = ConcurrentHashMap<String, MutableList<TimerSample>>()
    private val gauges = ConcurrentHashMap<String, AtomicLong>()
    private val histograms = ConcurrentHashMap<String, MutableList<Double>>()
    
    private val timersMutex = Mutex()
    private val histogramsMutex = Mutex()
    
    /**
     * Increment a counter metric
     * 
     * @param name Counter name
     * @param amount Amount to increment by
     */
    fun incrementCounter(name: String, amount: Long = 1) {
        counters.computeIfAbsent(name) { AtomicLong(0) }.addAndGet(amount)
    }

    /**
     * Set a gauge value
     * 
     * @param name Gauge name
     * @param value Value to set
     */
    fun setGauge(name: String, value: Long) {
        gauges.computeIfAbsent(name) { AtomicLong(0) }.set(value)
    }
    
    /**
     * Record a timer sample
     * 
     * @param name Timer name
     * @param durationMs Duration in milliseconds
     * @param tags Optional tags for the timer
     */
    suspend fun recordTimer(name: String, durationMs: Long, tags: Map<String, String> = emptyMap()) {
        val sample = TimerSample(durationMs, tags, Instant.now())
        timersMutex.withLock {
            timers.computeIfAbsent(name) { mutableListOf() }.add(sample)
        }
    }
    
    /**
     * Record a value in a histogram
     * 
     * @param name Histogram name
     * @param value Value to record
     */
    suspend fun recordHistogram(name: String, value: Double) {
        histogramsMutex.withLock {
            histograms.computeIfAbsent(name) { mutableListOf() }.add(value)
        }
    }
    
    /**
     * Get the value of a counter
     * 
     * @param name Counter name
     * @return Current value
     */
    fun getCounter(name: String): Long {
        return counters[name]?.get() ?: 0
    }
    
    /**
     * Get the value of a gauge
     * 
     * @param name Gauge name
     * @return Current value
     */
    fun getGauge(name: String): Long {
        return gauges[name]?.get() ?: 0
    }
    
    /**
     * Get timer statistics for a named timer
     * 
     * @param name Timer name
     * @return TimerStats with timer statistics or null if timer doesn't exist
     */
    suspend fun getTimerStats(name: String): TimerStats? {
        return timersMutex.withLock {
            val samples = timers[name] ?: return null
            if (samples.isEmpty()) return null
            
            val values = samples.map { it.durationMs.toDouble() }
            
            TimerStats(
                count = samples.size,
                min = values.minOrNull() ?: 0.0,
                max = values.maxOrNull() ?: 0.0,
                mean = values.average(),
                p50 = percentile(values, 50.0),
                p90 = percentile(values, 90.0),
                p95 = percentile(values, 95.0),
                p99 = percentile(values, 99.0)
            )
        }
    }
    
    /**
     * Get all metrics as a formatted string
     * 
     * @return Formatted metrics report
     */
    suspend fun getMetricsReport(): String {
        val report = StringBuilder()
        
        report.appendLine("# RAG Metrics Report - ${Instant.now()}")
        report.appendLine()
        
        // Counters
        if (counters.isNotEmpty()) {
            report.appendLine("## Counters")
            counters.forEach { (name, value) ->
                report.appendLine("$name: ${value.get()}")
            }
            report.appendLine()
        }
        
        // Gauges
        if (gauges.isNotEmpty()) {
            report.appendLine("## Gauges")
            gauges.forEach { (name, value) ->
                report.appendLine("$name: ${value.get()}")
            }
            report.appendLine()
        }
        
        // Timers
        timersMutex.withLock {
            if (timers.isNotEmpty()) {
                report.appendLine("## Timers")
                timers.forEach { (name, samples) ->
                    if (samples.isNotEmpty()) {
                        val values = samples.map { it.durationMs.toDouble() }
                        report.appendLine("$name:")
                        report.appendLine("  count: ${samples.size}")
                        report.appendLine("  min: ${values.minOrNull()?.toInt() ?: 0} ms")
                        report.appendLine("  max: ${values.maxOrNull()?.toInt() ?: 0} ms")
                        report.appendLine("  mean: ${values.average().toInt()} ms")
                        report.appendLine("  p50: ${percentile(values, 50.0).toInt()} ms")
                        report.appendLine("  p90: ${percentile(values, 90.0).toInt()} ms")
                        report.appendLine("  p95: ${percentile(values, 95.0).toInt()} ms")
                        report.appendLine("  p99: ${percentile(values, 99.0).toInt()} ms")
                    }
                }
                report.appendLine()
            }
        }
        
        // Histograms
        histogramsMutex.withLock {
            if (histograms.isNotEmpty()) {
                report.appendLine("## Histograms")
                histograms.forEach { (name, values) ->
                    if (values.isNotEmpty()) {
                        report.appendLine("$name:")
                        report.appendLine("  count: ${values.size}")
                        report.appendLine("  min: ${values.minOrNull() ?: 0.0}")
                        report.appendLine("  max: ${values.maxOrNull() ?: 0.0}")
                        report.appendLine("  mean: ${values.average()}")
                        report.appendLine("  p50: ${percentile(values, 50.0)}")
                        report.appendLine("  p90: ${percentile(values, 90.0)}")
                        report.appendLine("  p95: ${percentile(values, 95.0)}")
                        report.appendLine("  p99: ${percentile(values, 99.0)}")
                    }
                }
            }
        }
        
        return report.toString()
    }
    
    /**
     * Reset all metrics
     */
    suspend fun reset() {
        counters.clear()
        gauges.clear()
        timersMutex.withLock { timers.clear() }
        histogramsMutex.withLock { histograms.clear() }
    }
    
    /**
     * Calculate a percentile from a list of values
     */
    private fun percentile(values: List<Double>, percentile: Double): Double {
        if (values.isEmpty()) return 0.0
        
        val sorted = values.sorted()
        val index = (percentile / 100.0 * (sorted.size - 1)).toInt()
        return sorted[index]
    }
    
    /**
     * Data class for a timer sample
     */
    data class TimerSample(
        val durationMs: Long,
        val tags: Map<String, String> = emptyMap(),
        val timestamp: Instant = Instant.now()
    )
    
    /**
     * Data class for timer statistics
     */
    data class TimerStats(
        val count: Int,
        val min: Double,
        val max: Double,
        val mean: Double,
        val p50: Double,
        val p90: Double,
        val p95: Double,
        val p99: Double
    )
}

/**
 * Singleton metrics manager for the RAG library
 */
object RAGMetricsManager {
    private val metrics = RAGMetrics()
    
    fun getMetrics(): RAGMetrics = metrics
}

/**
 * Utility class to time operations
 */
class TimerContext(
    private val name: String,
    private val metrics: RAGMetrics,
    private val tags: Map<String, String> = emptyMap()
) {
    private val startTime = System.currentTimeMillis()
    
    /**
     * Stop timing and record the duration
     */
    suspend fun stop() {
        val duration = System.currentTimeMillis() - startTime
        metrics.recordTimer(name, duration, tags)
    }
}

/**
 * Extension function to time an operation
 */
suspend fun <T> RAGMetrics.time(name: String, tags: Map<String, String> = emptyMap(), block: suspend () -> T): T {
    val start = System.currentTimeMillis()
    try {
        return block()
    } finally {
        val duration = System.currentTimeMillis() - start
        recordTimer(name, duration, tags)
    }
}
