package com.gazapps.rag.core.error

import com.gazapps.rag.core.Embedder
import com.gazapps.rag.core.LLMClient
import com.gazapps.rag.core.VectorStore
import com.gazapps.rag.core.monitoring.RAGMetrics
import io.mockk.*
import kotlinx.coroutines.runBlocking
import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.assertThrows
import kotlin.test.assertEquals

class ErrorHandlingStrategyTest {

    private lateinit var mockEmbedder: Embedder
    private lateinit var mockFallbackEmbedder: Embedder
    private lateinit var mockVectorStore: VectorStore
    private lateinit var mockFallbackVectorStore: VectorStore
    private lateinit var mockLLMClient: LLMClient
    private lateinit var mockFallbackLLMClient: LLMClient
    private lateinit var mockLogger: RAGLogger
    private lateinit var mockMetrics: RAGMetrics
    private lateinit var mockCircuitBreaker: CircuitBreaker
    private lateinit var mockErrorHandler: ErrorHandler
    private lateinit var strategy: ErrorHandlingStrategy

    @BeforeEach
    fun setup() {
        mockEmbedder = mockk()
        mockFallbackEmbedder = mockk()
        mockVectorStore = mockk()
        mockFallbackVectorStore = mockk()
        mockLLMClient = mockk()
        mockFallbackLLMClient = mockk()
        mockLogger = mockk(relaxed = true)
        mockMetrics = mockk(relaxed = true)
        mockCircuitBreaker = mockk()
        mockErrorHandler = mockk()

        // Setup circuit breaker to just execute the block
        coEvery { mockCircuitBreaker.execute<Any>(any()) } answers { callOriginal() }
        
        // Setup error handler methods
        coEvery { mockErrorHandler.withCircuitBreaker<Any>(any(), any()) } answers { 
            secondArg<suspend () -> Any>().invoke()
        }
        coEvery { mockErrorHandler.withRetry<Any>(any(), any()) } answers { 
            secondArg<suspend () -> Any>().invoke()
        }
        coEvery { mockErrorHandler.withFallback<Any>(any(), any()) } answers { 
            firstArg<suspend () -> Any>().invoke()
        }

        strategy = ErrorHandlingStrategy(
            componentName = "TestRAG",
            logger = mockLogger,
            metrics = mockMetrics,
            errorHandler = mockErrorHandler,
            embeddingCircuitBreaker = mockCircuitBreaker,
            vectorStoreCircuitBreaker = mockCircuitBreaker,
            llmCircuitBreaker = mockCircuitBreaker
        )
    }

    @AfterEach
    fun tearDown() {
        clearAllMocks()
    }

    @Test
    fun `executeEmbedding should use primary embedder when it works`() = runBlocking {
        // Given
        val input = "test query"
        val expected = floatArrayOf(1f, 2f, 3f)
        coEvery { mockEmbedder.embed(input) } returns expected
        
        // When
        val result = strategy.executeEmbedding(
            primary = mockEmbedder,
            fallback = mockFallbackEmbedder,
            operation = { it.embed(input) },
            metricsKey = "test.embedding",
            logContext = mapOf("test" to "value")
        )
        
        // Then
        assertEquals(expected, result)
        coVerify { mockEmbedder.embed(input) }
        coVerify(exactly = 0) { mockFallbackEmbedder.embed(any()) }
        verify { mockMetrics.incrementCounter("test.embedding.attempts") }
    }

    @Test
    fun `executeEmbedding should use fallback embedder when primary fails`() = runBlocking {
        // Given
        val input = "test query"
        val expected = floatArrayOf(4f, 5f, 6f)
        val exception = RuntimeException("Primary embedder failed")
        
        // Set up primary to fail and fallback to succeed
        coEvery { mockEmbedder.embed(input) } throws exception
        coEvery { mockFallbackEmbedder.embed(input) } returns expected
        
        // Override error handler to simulate fallback behavior
        coEvery { mockErrorHandler.withFallback<FloatArray>(any(), any()) } answers { 
            secondArg<suspend (Exception) -> FloatArray>().invoke(exception)
        }
        
        // When
        val result = strategy.executeEmbedding(
            primary = mockEmbedder,
            fallback = mockFallbackEmbedder,
            operation = { it.embed(input) },
            metricsKey = "test.embedding",
            logContext = mapOf("test" to "value")
        )
        
        // Then
        assertEquals(expected, result)
        coVerify { mockEmbedder.embed(input) }
        coVerify { mockFallbackEmbedder.embed(input) }
        verify { mockLogger.warn(any(), any(), any(), any()) }
    }

    @Test
    fun `executeEmbedding should throw exception when both primary and fallback fail`() = runBlocking {
        // Given
        val input = "test query"
        val primaryException = RuntimeException("Primary embedder failed")
        val fallbackException = RuntimeException("Fallback embedder failed")
        
        // Set up both to fail
        coEvery { mockEmbedder.embed(input) } throws primaryException
        coEvery { mockFallbackEmbedder.embed(input) } throws fallbackException
        
        // Override error handler to simulate fallback failure
        coEvery { mockErrorHandler.withFallback<FloatArray>(any(), any()) } answers { 
            secondArg<suspend (Exception) -> FloatArray>().invoke(primaryException)
        }
        
        // When/Then
        assertThrows<RAGException.EmbeddingException> {
            runBlocking {
                strategy.executeEmbedding(
                    primary = mockEmbedder,
                    fallback = mockFallbackEmbedder,
                    operation = { it.embed(input) },
                    metricsKey = "test.embedding",
                    logContext = mapOf("test" to "value")
                )
            }
        }
        
        verify { mockMetrics.incrementCounter("test.embedding.attempts") }
        verify { mockMetrics.incrementCounter("test.embedding.failures") }
    }

    @Test
    fun `executeVectorStore should work with primary store`() = runBlocking {
        // Given
        val queryVector = floatArrayOf(1f, 2f, 3f)
        val expected = emptyList<Any>()
        
        coEvery { mockVectorStore.search(queryVector, any(), any()) } returns expected
        
        // When
        val result = strategy.executeVectorStore(
            primary = mockVectorStore,
            fallback = mockFallbackVectorStore,
            operation = { it.search(queryVector, 5) },
            operationName = "search",
            metricsKey = "test.vectorstore",
            logContext = mapOf("test" to "value")
        )
        
        // Then
        assertEquals(expected, result)
        coVerify { mockVectorStore.search(queryVector, 5, null) }
        coVerify(exactly = 0) { mockFallbackVectorStore.search(any(), any(), any()) }
        verify { mockMetrics.incrementCounter("test.vectorstore.attempts") }
    }

    @Test
    fun `executeLLM should work with primary LLM`() = runBlocking {
        // Given
        val prompt = "Generate text about cats"
        val expected = "Cats are wonderful pets that purr and meow."
        
        coEvery { mockLLMClient.generate(prompt) } returns expected
        
        // When
        val result = strategy.executeLLM(
            primary = mockLLMClient,
            fallback = mockFallbackLLMClient,
            operation = { it.generate(prompt) },
            metricsKey = "test.llm",
            logContext = mapOf("test" to "value")
        )
        
        // Then
        assertEquals(expected, result)
        coVerify { mockLLMClient.generate(prompt) }
        coVerify(exactly = 0) { mockFallbackLLMClient.generate(any()) }
        verify { mockMetrics.incrementCounter("test.llm.attempts") }
    }

    @Test
    fun `executeOperation should work for generic operations`() = runBlocking {
        // Given
        val expected = "Operation result"
        
        // When
        val result = strategy.executeOperation(
            operation = { expected },
            metricsKey = "test.generic",
            logContext = mapOf("test" to "value")
        )
        
        // Then
        assertEquals(expected, result)
        verify { mockMetrics.incrementCounter("test.generic.attempts") }
    }

    @Test
    fun `executeOperation should wrap exceptions in custom exception type`() = runBlocking {
        // Given
        val exception = RuntimeException("Operation failed")
        
        // When/Then
        val customException = assertThrows<RAGException.DocumentProcessingException> {
            runBlocking {
                strategy.executeOperation(
                    operation = { throw exception },
                    metricsKey = "test.generic",
                    exceptionMapper = { e -> RAGException.DocumentProcessingException("Custom message", e) },
                    logContext = mapOf("test" to "value")
                )
            }
        }
        
        assertEquals("Custom message", customException.message)
        verify { mockMetrics.incrementCounter("test.generic.attempts") }
        verify { mockMetrics.incrementCounter("test.generic.failures") }
        verify { mockLogger.error(any(), any(), any(), any()) }
    }
}
