package com.gazapps.rag.core.embedder

import com.gazapps.rag.core.Embedder
import com.gazapps.rag.core.VectorUtils
import io.ktor.client.*
import io.ktor.client.call.*
import io.ktor.client.engine.cio.*
import io.ktor.client.plugins.contentnegotiation.*
import io.ktor.client.request.*
import io.ktor.http.*
import io.ktor.serialization.kotlinx.json.*
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.Json
import org.slf4j.LoggerFactory

/**
 * Implementation of Embedder that uses OpenAI API to generate embeddings.
 *
 * @param apiKey The OpenAI API key
 * @param model The embedding model to use (default: "text-embedding-ada-002")
 * @param apiBaseUrl The base URL for the OpenAI API (default: "https://api.openai.com/v1")
 * @param normalize Whether to normalize the embeddings to unit length (default: true)
 * @param httpClient The HTTP client to use (default: a new client with JSON content negotiation)
 */
class OpenAIEmbedder(
    private val apiKey: String,
    private val model: String = "text-embedding-ada-002",
    private val apiBaseUrl: String = "https://api.openai.com/v1",
    private val normalize: Boolean = true,
    private val httpClient: HttpClient = HttpClient(CIO) {
        install(ContentNegotiation) {
            json(Json {
                ignoreUnknownKeys = true
                prettyPrint = false
                isLenient = true
            })
        }
    }
) : Embedder {
    
    private val logger = LoggerFactory.getLogger(OpenAIEmbedder::class.java)

    override suspend fun embed(text: String): FloatArray {
        if (text.isBlank()) {
            logger.debug("Empty text provided, returning zero vector")
            return FloatArray(1536) // OpenAI's ada-002 produces 1536-dimensional vectors
        }
        
        try {
            val response = httpClient.post("$apiBaseUrl/embeddings") {
                contentType(ContentType.Application.Json)
                header("Authorization", "Bearer $apiKey")
                setBody(OpenAIEmbedRequest(model = model, input = JsonPrimitive(text)))
            }
            
            val openAIResponse: OpenAIEmbedResponse = response.body()
            
            if (openAIResponse.data.isEmpty()) {
                throw IllegalStateException("OpenAI returned empty embedding data")
            }
            
            val embedding = openAIResponse.data.first().embedding.toFloatArray()
            
            return if (normalize) {
                VectorUtils.normalize(embedding)
            } else {
                embedding
            }
        } catch (e: Exception) {
            logger.error("Error embedding text with OpenAI: ${e.message}", e)
            throw EmbeddingException("Failed to generate embedding via OpenAI API", e)
        }
    }

    override suspend fun batchEmbed(texts: List<String>): List<FloatArray> {
        if (texts.isEmpty()) {
            return emptyList()
        }
        
        if (texts.size == 1) {
            return listOf(embed(texts.first()))
        }
        
        try {
            // OpenAI supports batch embedding in a single request
            // Convert the list to a JsonArray
            val jsonArray = JsonArray(texts.map { JsonPrimitive(it) })
            
            val response = httpClient.post("$apiBaseUrl/embeddings") {
                contentType(ContentType.Application.Json)
                header("Authorization", "Bearer $apiKey")
                setBody(OpenAIEmbedRequest(model = model, input = jsonArray))
            }
            
            val openAIResponse: OpenAIEmbedResponse = response.body()
            
            // Sort the results by index to ensure ordered response
            val sortedEmbeddings = openAIResponse.data
                .sortedBy { it.index }
                .map { it.embedding.toFloatArray() }
                .map { if (normalize) VectorUtils.normalize(it) else it }
            
            if (sortedEmbeddings.size != texts.size) {
                throw IllegalStateException("Expected ${texts.size} embeddings, but received ${sortedEmbeddings.size}")
            }
            
            return sortedEmbeddings
        } catch (e: Exception) {
            logger.error("Error batch embedding texts with OpenAI: ${e.message}", e)
            throw EmbeddingException("Failed to generate batch embeddings via OpenAI API", e)
        }
    }
    
    /**
     * Closes the HTTP client when it's no longer needed
     */
    fun close() {
        httpClient.close()
    }
}

/**
 * Exception thrown when embedding generation fails
 */
class EmbeddingException(message: String, cause: Throwable? = null) : Exception(message, cause)

/**
 * Request body for OpenAI embeddings API
 */
@Serializable
private data class OpenAIEmbedRequest(
    val model: String,
    val input: JsonElement,
    @SerialName("encoding_format") val encodingFormat: String = "float"
)

/**
 * Response from OpenAI embeddings API
 */
@Serializable
private data class OpenAIEmbedResponse(
    val data: List<OpenAIEmbedData>,
    val model: String,
    val usage: OpenAIEmbedUsage
)

/**
 * Embedding data in OpenAI response
 */
@Serializable
private data class OpenAIEmbedData(
    val embedding: List<Float>,
    val index: Int,
    @SerialName("object") val objectType: String
)

/**
 * Usage information in OpenAI response
 */
@Serializable
private data class OpenAIEmbedUsage(
    @SerialName("prompt_tokens") val promptTokens: Int,
    @SerialName("total_tokens") val totalTokens: Int
)
