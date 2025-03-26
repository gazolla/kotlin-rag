package com.gazapps.rag.core.llm

import com.gazapps.rag.core.GenerationOptions
import com.gazapps.rag.core.LLMClient
import io.ktor.client.*
import io.ktor.client.engine.mock.*
import io.ktor.client.plugins.contentnegotiation.*
import io.ktor.http.*
import io.ktor.serialization.kotlinx.json.*
import kotlinx.coroutines.runBlocking
import kotlinx.serialization.json.Json
import org.junit.jupiter.api.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class LLMClientTest {

    @Test
    fun `MockLLMClient should generate text based on templates`() = runBlocking {
        val mockClient = MockLLMClient()
        
        val response = mockClient.generate("What is RAG?")
        
        assertTrue(response.contains("Retrieval-Augmented Generation"))
    }
    
    @Test
    fun `MockLLMClient should extract context and question`() = runBlocking {
        val mockClient = MockLLMClient()
        
        val prompt = """
            Based on the following context:
            
            Kotlin is a modern programming language that runs on the JVM.
            
            Answer the question: What is Kotlin?
        """.trimIndent()
        
        val response = mockClient.generate(prompt)
        
        assertTrue(response.contains("Kotlin is a modern programming language"))
    }
    
    @Test
    fun `PromptFormatter should correctly format contexts with documents`() {
        val formatter = PromptFormatter()
        
        val documents = listOf(
            createScoredDocument("doc1", "Content of document 1", 0.95f),
            createScoredDocument("doc2", "Content of document 2", 0.85f)
        )
        
        val formattedPrompt = formatter.format(
            PromptTemplate.RAG_QA,
            "What do the documents say?",
            documents
        )
        
        assertTrue(formattedPrompt.contains("Based on the following context"))
        assertTrue(formattedPrompt.contains("Document 1"))
        assertTrue(formattedPrompt.contains("Content of document 1"))
        assertTrue(formattedPrompt.contains("Document 2"))
        assertTrue(formattedPrompt.contains("Content of document 2"))
        assertTrue(formattedPrompt.contains("Answer the question: What do the documents say?"))
    }
    
    @Test
    fun `ConversationManager should manage conversation history`() {
        val manager = ConversationManager()
        val conversation = manager.createConversation("test-conv")
        
        conversation.addUserMessage("Hello")
        conversation.addAssistantMessage("Hi there")
        conversation.addUserMessage("How are you?")
        
        assertEquals(3, conversation.messages.size)
        assertEquals("Hello", conversation.messages[0].content)
        assertEquals("Hi there", conversation.messages[1].content)
        assertEquals("How are you?", conversation.messages[2].content)
        
        val prompt = conversation.formatHistoryAsPrompt()
        assertTrue(prompt.contains("user: Hello"))
        assertTrue(prompt.contains("assistant: Hi there"))
        assertTrue(prompt.contains("How are you?"))
    }
    
    @Test
    fun `OpenAIClient should build correct request format`() {
        val openAIClient = createMockOpenAIClient("""
            {
                "choices": [
                    {
                        "message": {
                            "content": "This is a test response from OpenAI"
                        }
                    }
                ]
            }
        """.trimIndent())
        
        runBlocking {
            val response = openAIClient.generate("Test prompt")
            assertEquals("This is a test response from OpenAI", response)
        }
    }
    
    @Test
    fun `AnthropicClient should build correct request format`() {
        val anthropicClient = createMockAnthropicClient("""
            {
                "content": [
                    {
                        "type": "text",
                        "text": "This is a test response from Anthropic"
                    }
                ]
            }
        """.trimIndent())
        
        runBlocking {
            val response = anthropicClient.generate("Test prompt")
            assertEquals("This is a test response from Anthropic", response)
        }
    }
    
    // Helper functions for tests
    
    private fun createScoredDocument(id: String, content: String, score: Float): com.gazapps.rag.core.ScoredDocument {
        val document = com.gazapps.rag.core.Document {
            this.id = id
            this.content = content
            this.metadata = mapOf("source" to "test")
        }
        return com.gazapps.rag.core.ScoredDocument(document, score)
    }
    
    private fun createMockOpenAIClient(responseJson: String): OpenAIClient {
        val mockEngine = MockEngine { request ->
            // Verify request URL and headers
            assertEquals("https://api.openai.com/v1/chat/completions", request.url.toString())
            assertTrue(request.headers.contains("Authorization"))
            
            respond(
                content = responseJson,
                status = HttpStatusCode.OK,
                headers = headersOf(HttpHeaders.ContentType, "application/json")
            )
        }
        
        val httpClient = HttpClient(mockEngine) {
            install(ContentNegotiation) {
                json(Json {
                    ignoreUnknownKeys = true
                    isLenient = true
                })
            }
        }
        
        return OpenAIClient(
            apiKey = "test-key",
            httpClient = httpClient
        )
    }
    
    private fun createMockAnthropicClient(responseJson: String): AnthropicClient {
        val mockEngine = MockEngine { request ->
            // Verify request URL and headers
            assertEquals("https://api.anthropic.com/v1/messages", request.url.toString())
            assertTrue(request.headers.contains("x-api-key"))
            assertTrue(request.headers.contains("anthropic-version"))
            
            respond(
                content = responseJson,
                status = HttpStatusCode.OK,
                headers = headersOf(HttpHeaders.ContentType, "application/json")
            )
        }
        
        val httpClient = HttpClient(mockEngine) {
            install(ContentNegotiation) {
                json(Json {
                    ignoreUnknownKeys = true
                    isLenient = true
                })
            }
        }
        
        return AnthropicClient(
            apiKey = "test-key",
            httpClient = httpClient
        )
    }
}
