package com.gazapps.rag.core.llm

import com.gazapps.rag.core.GenerationOptions
import com.gazapps.rag.core.LLMClient
import com.gazapps.rag.core.utils.RateLimiter
import io.ktor.client.*
import io.ktor.client.call.*
import io.ktor.client.plugins.contentnegotiation.*
import io.ktor.client.request.*
import io.ktor.client.statement.*
import io.ktor.http.*
import io.ktor.serialization.kotlinx.json.*
import kotlinx.serialization.json.*
import kotlinx.coroutines.CancellationException

/**
 * A client for the OpenAI API that implements the LLMClient interface
 *
 * @property apiKey Your OpenAI API key
 * @property model The model to use for text generation (default: "gpt-3.5-turbo")
 * @property apiBaseUrl The base URL for the OpenAI API (default: "https://api.openai.com/v1")
 * @property httpClient The HTTP client to use for requests
 * @property maxRetries Number of retries for failed requests
 * @property requestsPerMinute Rate limit for API requests
 */
class OpenAIClient(
    private val apiKey: String,
    private val model: String = "gpt-3.5-turbo",
    private val apiBaseUrl: String = "https://api.openai.com/v1",
    private val httpClient: HttpClient = defaultHttpClient(),
    private val maxRetries: Int = 3,
    requestsPerMinute: Int = 60
) : LLMClient {
    private val rateLimiter = RateLimiter(requestsPerMinute)
    
    /**
     * Generate text using the default generation options
     */
    override suspend fun generate(prompt: String): String {
        return generate(prompt, GenerationOptions())
    }
    
    /**
     * Generate text with custom generation options
     */
    override suspend fun generate(prompt: String, options: GenerationOptions): String {
        rateLimiter.acquire()
        
        var attempt = 0
        var lastException: Exception? = null
        
        while (attempt < maxRetries) {
            try {
                val response = httpClient.post("$apiBaseUrl/chat/completions") {
                    contentType(ContentType.Application.Json)
                    header("Authorization", "Bearer $apiKey")
                    setBody(buildRequestBody(prompt, options))
                }
                
                if (response.status.isSuccess()) {
                    return parseResponse(response.body())
                } else {
                    val errorBody = response.bodyAsText()
                    throw Exception("API request failed with status ${response.status}: $errorBody")
                }
            } catch (e: Exception) {
                if (e is CancellationException) {
                    throw e
                }
                
                lastException = e
                
                // Exponential backoff
                val backoffMs = (2.0.pow(attempt.toDouble()) * 1000).toLong()
                kotlinx.coroutines.delay(backoffMs)
                attempt++
            }
        }
        
        throw lastException ?: Exception("Failed to generate text after $maxRetries attempts")
    }
    
    /**
     * Build the JSON request body for the OpenAI API
     */
    private fun buildRequestBody(prompt: String, options: GenerationOptions): JsonObject {
        return buildJsonObject {
            put("model", model)
            put("messages", buildJsonArray {
                add(buildJsonObject {
                    put("role", "user")
                    put("content", prompt)
                })
            })
            put("temperature", options.temperature)
            put("top_p", options.topP)
            put("max_tokens", options.maxTokens)
            
            if (options.stop.isNotEmpty()) {
                put("stop", buildJsonArray {
                    options.stop.forEach { add(it) }
                })
            }
        }
    }
    
    /**
     * Parse the response from the OpenAI API
     */
    private fun parseResponse(json: JsonElement): String {
        return try {
            val choices = json.jsonObject["choices"]?.jsonArray
            val firstChoice = choices?.getOrNull(0)?.jsonObject
            val message = firstChoice?.get("message")?.jsonObject
            val content = message?.get("content")?.jsonPrimitive?.content
            
            content ?: throw Exception("Failed to parse response: $json")
        } catch (e: Exception) {
            throw Exception("Failed to parse OpenAI response: ${e.message}")
        }
    }
    
    /**
     * Create a default HTTP client
     */
    companion object {
        private fun defaultHttpClient() = HttpClient {
            install(ContentNegotiation) {
                json(Json {
                    ignoreUnknownKeys = true
                    isLenient = true
                    prettyPrint = false
                })
            }
        }
    }
}

private fun Double.pow(exponent: Double): Double = Math.pow(this, exponent)
