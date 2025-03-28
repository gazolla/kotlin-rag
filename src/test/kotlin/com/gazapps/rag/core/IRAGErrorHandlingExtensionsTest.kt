package com.gazapps.rag.core

import com.gazapps.rag.core.config.RAGConfig
import com.gazapps.rag.core.error.*
import com.gazapps.rag.core.monitoring.RAGMetrics
import io.mockk.*
import kotlinx.coroutines.runBlocking
import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import kotlin.test.assertEquals

class IRAGErrorHandlingExtensionsTest {

    private lateinit var mockRAG: IRAG
    private lateinit var mockEmbedder: Embedder
    private lateinit var mockVectorStore: VectorStore
    private lateinit var mockLLMClient: LLMClient
    private lateinit var mockLogger: RAGLogger
    private lateinit var mockMetrics: RAGMetrics
    private lateinit var mockErrorHandlingStrategy: ErrorHandlingStrategy

    @BeforeEach
    fun setup() {
        mockRAG = mockk()
        mockEmbedder = mockk()
        mockVectorStore = mockk()
        mockLLMClient = mockk()
        mockLogger = mockk(relaxed = true)
        mockMetrics = mockk(relaxed = true)
        mockErrorHandlingStrategy = mockk()
        
        // Setup IRAG implementation getters
        every { mockRAG.getCurrentEmbedder() } returns mockEmbedder
        every { mockRAG.getCurrentVectorStore() } returns mockVectorStore
        every { mockRAG.getCurrentLLMClient() } returns mockLLMClient
        every { mockRAG.getLogger() } returns mockLogger
        every { mockRAG.getMetrics() } returns mockMetrics
        every { mockRAG.getFallbackEmbedder() } returns null
        every { mockRAG.getFallbackVectorStore() } returns null
        every { mockRAG.getFallbackLLMClient() } returns null
        every { mockRAG.getCurrentConfig() } returns RAGConfig()
    }

    @AfterEach
    fun tearDown() {
        clearAllMocks()
    }

    @Test
    fun `executeWithErrorHandling should delegate to error handling strategy for embedder operations`() = runBlocking {
        // Given
        val input = "test query"
        val expected = floatArrayOf(1f, 2f, 3f)
        
        coEvery { 
            mockErrorHandlingStrategy.executeEmbedding(
                any(), any(), any(), any(), any()
            ) 
        } answers {
            val operation = arg<suspend (Embedder) -> FloatArray>(2)
            operation(mockEmbedder)
        }
        
        coEvery { mockEmbedder.embed(input) } returns expected
        
        // When
        val result = mockRAG.executeWithErrorHandling(
            operation = { embedder -> embedder.embed(input) },
            metricsKey = "test.embedding",
            logContext = mapOf("test" to "value"),
            errorHandlingStrategy = mockErrorHandlingStrategy
        )
        
        // Then
        assertEquals(expected, result)
        coVerify { mockEmbedder.embed(input) }
        coVerify { 
            mockErrorHandlingStrategy.executeEmbedding(
                primary = mockEmbedder,
                fallback = null,
                operation = any(),
                metricsKey = "test.embedding",
                logContext = mapOf("test" to "value")
            ) 
        }
    }

    @Test
    fun `executeVectorStoreWithErrorHandling should delegate to error handling strategy`() = runBlocking {
        // Given
        val queryVector = floatArrayOf(1f, 2f, 3f)
        val expected = emptyList<ScoredDocument>()
        
        coEvery { 
            mockErrorHandlingStrategy.executeVectorStore(
                any(), any(), any(), any(), any(), any()
            ) 
        } answers {
            val operation = arg<suspend (VectorStore) -> List<ScoredDocument>>(2)
            operation(mockVectorStore)
        }
        
        coEvery { mockVectorStore.search(queryVector, any(), any()) } returns expected
        
        // When
        val result = mockRAG.executeVectorStoreWithErrorHandling(
            operation = { vectorStore -> vectorStore.search(queryVector, 5) },
            operationName = "search",
            metricsKey = "test.vectorstore",
            logContext = mapOf("test" to "value"),
            errorHandlingStrategy = mockErrorHandlingStrategy
        )
        
        // Then
        assertEquals(expected, result)
        coVerify { mockVectorStore.search(queryVector, 5, null) }
        coVerify { 
            mockErrorHandlingStrategy.executeVectorStore(
                primary = mockVectorStore,
                fallback = null,
                operation = any(),
                operationName = "search",
                metricsKey = "test.vectorstore",
                logContext = mapOf("test" to "value")
            ) 
        }
    }

    @Test
    fun `executeLLMWithErrorHandling should delegate to error handling strategy`() = runBlocking {
        // Given
        val prompt = "Generate text about cats"
        val expected = "Cats are wonderful pets that purr and meow."
        
        coEvery { 
            mockErrorHandlingStrategy.executeLLM(
                any(), any(), any(), any(), any()
            ) 
        } answers {
            val operation = arg<suspend (LLMClient) -> String>(2)
            operation(mockLLMClient)
        }
        
        coEvery { mockLLMClient.generate(prompt) } returns expected
        
        // When
        val result = mockRAG.executeLLMWithErrorHandling(
            operation = { llmClient -> llmClient.generate(prompt) },
            metricsKey = "test.llm",
            logContext = mapOf("test" to "value"),
            errorHandlingStrategy = mockErrorHandlingStrategy
        )
        
        // Then
        assertEquals(expected, result)
        coVerify { mockLLMClient.generate(prompt) }
        coVerify { 
            mockErrorHandlingStrategy.executeLLM(
                primary = mockLLMClient,
                fallback = null,
                operation = any(),
                metricsKey = "test.llm",
                logContext = mapOf("test" to "value")
            ) 
        }
    }

    @Test
    fun `queryWithErrorHandling should compose multiple error handling calls`() = runBlocking {
        // Given
        val question = "What is RAG?"
        val embedding = floatArrayOf(1f, 2f, 3f)
        val documents = listOf<ScoredDocument>()
        val expectedAnswer = "RAG is Retrieval-Augmented Generation"
        
        // Setup the error handling strategy to handle the full flow
        coEvery { 
            mockErrorHandlingStrategy.executeEmbedding(any(), any(), any(), any(), any())
        } returns embedding
        
        coEvery { 
            mockErrorHandlingStrategy.executeVectorStore(any(), any(), any(), any(), any(), any())
        } returns documents
        
        coEvery { 
            mockErrorHandlingStrategy.executeLLM(any(), any(), any(), any(), any())
        } returns expectedAnswer
        
        // When
        val result = mockRAG.queryWithErrorHandling(
            question = question,
            errorHandlingStrategy = mockErrorHandlingStrategy
        )
        
        // Then
        assertEquals(expectedAnswer, result.answer)
        assertEquals(documents, result.documents)
        
        coVerify { 
            mockErrorHandlingStrategy.executeEmbedding(
                primary = mockEmbedder,
                fallback = null,
                operation = any(),
                metricsKey = "rag.query.embedding", 
                logContext = any()
            )
        }
        
        coVerify { 
            mockErrorHandlingStrategy.executeVectorStore(
                primary = mockVectorStore,
                fallback = null,
                operation = any(),
                operationName = "search",
                metricsKey = "rag.query.retrieval",
                logContext = any()
            )
        }
        
        coVerify { 
            mockErrorHandlingStrategy.executeLLM(
                primary = mockLLMClient,
                fallback = null,
                operation = any(),
                metricsKey = "rag.query.generation",
                logContext = any()
            )
        }
    }
}
