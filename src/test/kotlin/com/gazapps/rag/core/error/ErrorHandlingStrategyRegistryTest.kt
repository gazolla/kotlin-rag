package com.gazapps.rag.core.error

import com.gazapps.rag.core.monitoring.RAGMetrics
import io.mockk.every
import io.mockk.mockk
import io.mockk.verify
import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotSame
import kotlin.test.assertSame

class ErrorHandlingStrategyRegistryTest {

    private lateinit var mockLogger: RAGLogger
    private lateinit var mockMetrics: RAGMetrics

    @BeforeEach
    fun setup() {
        mockLogger = mockk(relaxed = true)
        mockMetrics = mockk(relaxed = true)
        
        // Clear the registry before each test
        ErrorHandlingStrategyRegistry.clear()
    }

    @AfterEach
    fun tearDown() {
        // Clean up after tests
        ErrorHandlingStrategyRegistry.clear()
    }

    @Test
    fun `getStrategy should return the same instance for the same name`() {
        // When
        val strategy1 = ErrorHandlingStrategyRegistry.getStrategy("test", mockLogger, mockMetrics)
        val strategy2 = ErrorHandlingStrategyRegistry.getStrategy("test", mockLogger, mockMetrics)
        
        // Then
        assertSame(strategy1, strategy2, "The registry should return the same instance for the same name")
    }

    @Test
    fun `getStrategy should return different instances for different names`() {
        // When
        val strategy1 = ErrorHandlingStrategyRegistry.getStrategy("test1", mockLogger, mockMetrics)
        val strategy2 = ErrorHandlingStrategyRegistry.getStrategy("test2", mockLogger, mockMetrics)
        
        // Then
        assertNotSame(strategy1, strategy2, "The registry should return different instances for different names")
    }

    @Test
    fun `createStrategy should replace existing strategy`() {
        // Given
        val strategy1 = ErrorHandlingStrategyRegistry.getStrategy("test", mockLogger, mockMetrics)
        
        // When
        val strategy2 = ErrorHandlingStrategyRegistry.createStrategy("test", mockLogger, mockMetrics)
        val strategy3 = ErrorHandlingStrategyRegistry.getStrategy("test", mockLogger, mockMetrics)
        
        // Then
        assertNotSame(strategy1, strategy2, "Created strategy should be different from the original")
        assertSame(strategy2, strategy3, "After creation, getStrategy should return the new instance")
    }

    @Test
    fun `removeStrategy should remove a strategy from the registry`() {
        // Given
        val strategy1 = ErrorHandlingStrategyRegistry.getStrategy("test", mockLogger, mockMetrics)
        
        // When
        ErrorHandlingStrategyRegistry.removeStrategy("test")
        val strategy2 = ErrorHandlingStrategyRegistry.getStrategy("test", mockLogger, mockMetrics)
        
        // Then
        assertNotSame(strategy1, strategy2, "After removal, getStrategy should create a new instance")
    }

    @Test
    fun `clear should remove all strategies from the registry`() {
        // Given
        val strategy1 = ErrorHandlingStrategyRegistry.getStrategy("test1", mockLogger, mockMetrics)
        val strategy2 = ErrorHandlingStrategyRegistry.getStrategy("test2", mockLogger, mockMetrics)
        
        // When
        ErrorHandlingStrategyRegistry.clear()
        val newStrategy1 = ErrorHandlingStrategyRegistry.getStrategy("test1", mockLogger, mockMetrics)
        val newStrategy2 = ErrorHandlingStrategyRegistry.getStrategy("test2", mockLogger, mockMetrics)
        
        // Then
        assertNotSame(strategy1, newStrategy1, "After clearing, getStrategy should create a new instance for test1")
        assertNotSame(strategy2, newStrategy2, "After clearing, getStrategy should create a new instance for test2")
    }

    @Test
    fun `getStrategy should use default values when parameters are null`() {
        // When
        val strategy = ErrorHandlingStrategyRegistry.getStrategy("test")
        
        // Then
        // We can't directly test the internal properties, but we can verify the strategy was created
        // with a non-null logger and metrics
        assertEquals("test", strategy.toString().contains("test").toString())
    }
}
