package com.gazapps.rag.core.monitoring

import kotlinx.coroutines.runBlocking
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test

class RAGMetricsTest {
    
    private lateinit var metrics: RAGMetrics
    
    @BeforeEach
    fun setup() = runBlocking {
        metrics = RAGMetrics()
        metrics.reset()
    }
    
    @Test
    fun `counter metrics should increment correctly`() {
        assertEquals(0, metrics.getCounter("test-counter"))
        
        metrics.incrementCounter("test-counter")
        assertEquals(1, metrics.getCounter("test-counter"))
        
        metrics.incrementCounter("test-counter", 5)
        assertEquals(6, metrics.getCounter("test-counter"))
    }
    
    @Test
    fun `gauge metrics should set correctly`() {
        assertEquals(0, metrics.getGauge("test-gauge"))
        
        metrics.setGauge("test-gauge", 42)
        assertEquals(42, metrics.getGauge("test-gauge"))
        
        metrics.setGauge("test-gauge", 100)
        assertEquals(100, metrics.getGauge("test-gauge"))
    }
    
    @Test
    fun `timer metrics should record and calculate statistics`() = runBlocking {
        // Record some timers
        metrics.recordTimer("test-timer", 100)
        metrics.recordTimer("test-timer", 200)
        metrics.recordTimer("test-timer", 300)
        metrics.recordTimer("test-timer", 400)
        metrics.recordTimer("test-timer", 500)
        
        // Get stats
        val stats = metrics.getTimerStats("test-timer")
        
        // Verify stats
        assertNotNull(stats)
        assertEquals(5, stats!!.count)
        assertEquals(100.0, stats.min, 0.0)
        assertEquals(500.0, stats.max, 0.0)
        assertEquals(300.0, stats.mean, 0.0)
        assertEquals(300.0, stats.p50, 0.0)
        assertEquals(450.0, stats.p90, 0.0)
        assertEquals(500.0, stats.p95, 0.0)
        assertEquals(500.0, stats.p99, 0.0)
    }
    
    @Test
    fun `histogram metrics should record and calculate percentiles`() = runBlocking {
        // Record some values
        for (i in 1..100) {
            metrics.recordHistogram("test-histogram", i.toDouble())
        }
        
        // Get metrics report
        val report = metrics.getMetricsReport()
        
        // Verify report contains histogram data
        assertTrue(report.contains("test-histogram"))
        assertTrue(report.contains("count: 100"))
        assertTrue(report.contains("min: 1.0"))
        assertTrue(report.contains("max: 100.0"))
        
        // Should have approximately the right percentiles
        assertTrue(report.contains("p50: 50.0") || report.contains("p50: 51.0"))
        assertTrue(report.contains("p90: 90.0") || report.contains("p90: 91.0"))
        assertTrue(report.contains("p95: 95.0") || report.contains("p95: 96.0"))
    }
    
    @Test
    fun `time function should record duration of operations`() = runBlocking {
        val result = metrics.time("timed-operation") {
            // Simulate work
            Thread.sleep(50)
            "result"
        }
        
        // Check result
        assertEquals("result", result)
        
        // Check timer was recorded
        val stats = metrics.getTimerStats("timed-operation")
        assertNotNull(stats)
        assertEquals(1, stats!!.count)
        assertTrue(stats.min >= 50)
    }
    
    @Test
    fun `time function should record duration even when operation throws`() = runBlocking {
        val exception = assertThrows(RuntimeException::class.java) {
            metrics.time("failing-operation") {
                Thread.sleep(50)
                throw RuntimeException("Test exception")
            }
        }
        
        assertEquals("Test exception", exception.message)
        
        // Check timer was still recorded
        val stats = metrics.getTimerStats("failing-operation")
        assertNotNull(stats)
        assertEquals(1, stats!!.count)
        assertTrue(stats.min >= 50)
    }
    
    @Test
    fun `metrics report should include all metric types`() = runBlocking {
        // Add various metrics
        metrics.incrementCounter("test-counter", 10)
        metrics.setGauge("test-gauge", 42)
        metrics.recordTimer("test-timer", 100)
        metrics.recordHistogram("test-histogram", 3.14)
        
        // Get report
        val report = metrics.getMetricsReport()
        
        // Verify report contains all metrics
        assertTrue(report.contains("test-counter: 10"))
        assertTrue(report.contains("test-gauge: 42"))
        assertTrue(report.contains("test-timer"))
        assertTrue(report.contains("test-histogram"))
    }
    
    @Test
    fun `reset should clear all metrics`() = runBlocking {
        // Add various metrics
        metrics.incrementCounter("test-counter", 10)
        metrics.setGauge("test-gauge", 42)
        metrics.recordTimer("test-timer", 100)
        metrics.recordHistogram("test-histogram", 3.14)
        
        // Reset
        metrics.reset()
        
        // Verify all metrics are reset
        assertEquals(0, metrics.getCounter("test-counter"))
        assertEquals(0, metrics.getGauge("test-gauge"))
        assertNull(metrics.getTimerStats("test-timer"))
        
        // Report should be mostly empty
        val report = metrics.getMetricsReport()
        assertFalse(report.contains("test-counter: 10"))
        assertFalse(report.contains("test-gauge: 42"))
        assertFalse(report.contains("count: 1")) // From timer or histogram
    }
}
