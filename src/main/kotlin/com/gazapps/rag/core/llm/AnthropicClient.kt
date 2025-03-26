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
 * A client for the Anthropic Claude API that implements the LLMClient interface
 *
 * @property apiKey Your Anthropic API key
 * @property model The model to use for text generation (default: "claude-3-sonnet-20240229")
 * @property apiBaseUrl The base URL for the Anthropic API (default: "https://api.anthropic.com/v1")
 * @property httpClient The HTTP client to use for requests
 * @property maxRetries Number of retries for failed requests
 * @property requestsPerMinute Rate limit for API requests
 */
class AnthropicClient(
    private val apiKey: String,
    private val model: String = "claude-3-sonnet-20240229",
    private val apiBaseUrl: String = "https://api.anthropic.com/v1",
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
                val response = httpClient.post("$apiBaseUrl/messages") {
                    contentType(ContentType.Application.Json)
                    header("x-api-key", apiKey)
                    header("anthropic-version", "2023-06-01")
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
     * Build the JSON request body for the Anthropic API
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
                put("stop_sequences", buildJsonArray {
                    options.stop.forEach { add(it) }
                })
            }
        }
    }
    
    /**
     * Parse the response from the Anthropic API
     */
    private fun parseResponse(json: JsonElement): String {
        return try {
            val content = json.jsonObject["content"]?.jsonArray
                ?.filter { it.jsonObject["type"]?.jsonPrimitive?.content == "text" }
                ?.mapNotNull { it.jsonObject["text"]?.jsonPrimitive?.content }
                ?.joinToString("") ?: ""
            
            if (content.isBlank()) {
                throw Exception("Empty response from Anthropic API: $json")
            }
            
            content
        } catch (e: Exception) {
            throw Exception("Failed to parse Anthropic response: ${e.message}")
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
