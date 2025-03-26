package com.gazapps.rag.core.embedder

import io.ktor.client.*
import io.ktor.client.engine.mock.*
import io.ktor.client.plugins.contentnegotiation.*
import io.ktor.http.*
import io.ktor.serialization.kotlinx.json.*
import kotlinx.coroutines.runBlocking
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

class ApiEmbedderTest {
    
    private val mockJson = Json { 
        ignoreUnknownKeys = true 
        prettyPrint = false
        isLenient = true
    }
    
    // Helper to create mock response for OpenAI
    private fun createOpenAIMockEngine(
        responseBody: String? = null,
        status: HttpStatusCode = HttpStatusCode.OK
    ): HttpClientEngine {
        return MockEngine { request ->
            if (request.url.toString().contains("embeddings")) {
                val responseContent = responseBody ?: """
                    {
                        "object": "list",
                        "data": [
                            {
                                "object": "embedding",
                                "embedding": [0.1, 0.2, 0.3, 0.4, 0.5],
                                "index": 0
                            }
                        ],
                        "model": "text-embedding-ada-002",
                        "usage": {
                            "prompt_tokens": 5,
                            "total_tokens": 5
                        }
                    }
                """.trimIndent()
                
                respond(
                    content = responseContent,
                    status = status,
                    headers = headersOf(HttpHeaders.ContentType, "application/json")
                )
            } else {
                error("Unhandled request: ${request.url}")
            }
        }
    }
    
    // Helper to create mock response for Hugging Face
    private fun createHuggingFaceMockEngine(
        responseBody: String? = null,
        status: HttpStatusCode = HttpStatusCode.OK
    ): HttpClientEngine {
        return MockEngine { request ->
            val responseContent = responseBody ?: """
                [[0.1, 0.2, 0.3, 0.4, 0.5]]
            """.trimIndent()
            
            respond(
                content = responseContent,
                status = status,
                headers = headersOf(HttpHeaders.ContentType, "application/json")
            )
        }
    }
    
    @Test
    fun `OpenAIEmbedder should handle successful response`() = runBlocking {
        // Setup
        val mockEngine = createOpenAIMockEngine()
        val mockClient = HttpClient(mockEngine) {
            install(ContentNegotiation) {
                json(mockJson)
            }
        }
        
        val embedder = OpenAIEmbedder(
            apiKey = "dummy-key",
            httpClient = mockClient
        )
        
        // Act
        val embedding = embedder.embed("test text")
        
        // Assert
        assertEquals(5, embedding.size)
        assertEquals(0.1f, embedding[0])
        assertEquals(0.5f, embedding[4])
    }
    
    @Test
    fun `OpenAIEmbedder should handle batch embedding`() = runBlocking {
        // Setup
        val batchResponse = """
            {
                "object": "list",
                "data": [
                    {
                        "object": "embedding",
                        "embedding": [0.1, 0.2, 0.3],
                        "index": 0
                    },
                    {
                        "object": "embedding",
                        "embedding": [0.4, 0.5, 0.6],
                        "index": 1
                    }
                ],
                "model": "text-embedding-ada-002",
                "usage": {
                    "prompt_tokens": 10,
                    "total_tokens": 10
                }
            }
        """.trimIndent()
        
        val mockEngine = createOpenAIMockEngine(batchResponse)
        val mockClient = HttpClient(mockEngine) {
            install(ContentNegotiation) {
                json(mockJson)
            }
        }
        
        val embedder = OpenAIEmbedder(
            apiKey = "dummy-key",
            httpClient = mockClient
        )
        
        // Act
        val embeddings = embedder.batchEmbed(listOf("text1", "text2"))
        
        // Assert
        assertEquals(2, embeddings.size)
        assertEquals(3, embeddings[0].size)
        assertEquals(0.1f, embeddings[0][0])
        assertEquals(0.6f, embeddings[1][2])
    }
    
    @Test
    fun `OpenAIEmbedder should handle error responses`() = runBlocking {
        // Setup
        val errorEngine = createOpenAIMockEngine(
            responseBody = """{"error": {"message": "Invalid API key", "type": "invalid_request_error"}}""",
            status = HttpStatusCode.Unauthorized
        )
        
        val mockClient = HttpClient(errorEngine) {
            install(ContentNegotiation) {
                json(mockJson)
            }
            expectSuccess = false
        }
        
        val embedder = OpenAIEmbedder(
            apiKey = "invalid-key",
            httpClient = mockClient
        )
        
        // Act & Assert
        assertFailsWith<EmbeddingException> {
            embedder.embed("test text")
        }
    }
    
    @Test
    fun `HuggingFaceEmbedder should handle successful response`() = runBlocking {
        // Setup
        val mockEngine = createHuggingFaceMockEngine()
        val mockClient = HttpClient(mockEngine) {
            install(ContentNegotiation) {
                json(mockJson)
            }
        }
        
        val embedder = HuggingFaceEmbedder(
            apiKey = "dummy-key",
            httpClient = mockClient
        )
        
        // Act
        val embedding = embedder.embed("test text")
        
        // Assert
        assertEquals(5, embedding.size)
        assertEquals(0.1f, embedding[0])
        assertEquals(0.5f, embedding[4])
    }
    
    @Test
    fun `HuggingFaceEmbedder should handle batch embedding`() = runBlocking {
        // Setup
        val batchResponse = """
            [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6]
            ]
        """.trimIndent()
        
        val mockEngine = createHuggingFaceMockEngine(batchResponse)
        val mockClient = HttpClient(mockEngine) {
            install(ContentNegotiation) {
                json(mockJson)
            }
        }
        
        val embedder = HuggingFaceEmbedder(
            apiKey = "dummy-key",
            httpClient = mockClient
        )
        
        // Act
        val embeddings = embedder.batchEmbed(listOf("text1", "text2"))
        
        // Assert
        assertEquals(2, embeddings.size)
        assertEquals(3, embeddings[0].size)
        assertEquals(0.1f, embeddings[0][0])
        assertEquals(0.6f, embeddings[1][2])
    }
}
