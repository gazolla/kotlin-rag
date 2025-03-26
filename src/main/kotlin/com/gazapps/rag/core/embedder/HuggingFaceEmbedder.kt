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
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.JsonArray
import org.slf4j.LoggerFactory

/**
 * Implementation of Embedder that uses Hugging Face's Inference API to generate embeddings.
 * This uses the feature-extraction pipeline which returns embeddings for input texts.
 *
 * @param apiKey The Hugging Face API token
 * @param model The model to use for embeddings (default: "sentence-transformers/all-MiniLM-L6-v2")
 * @param normalize Whether to normalize the embeddings to unit length (default: true)
 * @param batchSize Maximum number of texts to send in a single API call (default: 8)
 * @param httpClient The HTTP client to use (default: a new client with JSON content negotiation)
 */
class HuggingFaceEmbedder(
    private val apiKey: String,
    private val model: String = "sentence-transformers/all-MiniLM-L6-v2",
    private val normalize: Boolean = true,
    private val batchSize: Int = 8,
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
    
    private val logger = LoggerFactory.getLogger(HuggingFaceEmbedder::class.java)
    
    // Base URL for Hugging Face Inference API
    private val apiBaseUrl = "https://api-inference.huggingface.co/models/$model"

    override suspend fun embed(text: String): FloatArray {
        if (text.isBlank()) {
            logger.debug("Empty text provided, returning zero vector")
            // Most sentence transformer models produce 384-dimensional vectors
            // but we can't know for sure without a model query
            return FloatArray(384)
        }
        
        try {
            val response = httpClient.post(apiBaseUrl) {
                contentType(ContentType.Application.Json)
                header("Authorization", "Bearer $apiKey")
                setBody(HuggingFaceRequest(JsonPrimitive(text), options = HuggingFaceOptions(useCache = true, waitForModel = true)))
            }
            
            // Different models return different structures, but most return a list of lists
            // when a sentence is the input. The outer list is batch dimension, inner list is embedding.
            val embeddings: List<List<Float>> = response.body()
            
            if (embeddings.isEmpty() || embeddings.first().isEmpty()) {
                throw IllegalStateException("Hugging Face returned empty embedding data")
            }
            
            // Get the first embedding (should be only one for a single text input)
            val embedding = embeddings.first().toFloatArray()
            
            return if (normalize) {
                VectorUtils.normalize(embedding)
            } else {
                embedding
            }
        } catch (e: Exception) {
            logger.error("Error embedding text with Hugging Face: ${e.message}", e)
            throw EmbeddingException("Failed to generate embedding via Hugging Face API", e)
        }
    }

    override suspend fun batchEmbed(texts: List<String>): List<FloatArray> {
        if (texts.isEmpty()) {
            return emptyList()
        }
        
        if (texts.size == 1) {
            return listOf(embed(texts.first()))
        }
        
        // Process in batches according to batchSize to avoid overwhelmingly large requests
        val results = mutableListOf<FloatArray>()
        
        for (i in texts.indices step batchSize) {
            val batch = texts.subList(i, minOf(i + batchSize, texts.size))
            results.addAll(processBatch(batch))
        }
        
        return results
    }
    
    /**
     * Process a batch of texts to get embeddings
     */
    private suspend fun processBatch(texts: List<String>): List<FloatArray> {
        try {
            // Convert the list to a JsonArray
            val jsonArray = JsonArray(texts.map { JsonPrimitive(it) })
            
            val response = httpClient.post(apiBaseUrl) {
                contentType(ContentType.Application.Json)
                header("Authorization", "Bearer $apiKey")
                setBody(HuggingFaceRequest(
                    inputs = jsonArray,
                    options = HuggingFaceOptions(useCache = true, waitForModel = true)
                ))
            }
            
            // For batch inputs, the API typically returns a list of embeddings
            val embeddings: List<List<Float>> = response.body()
            
            if (embeddings.size != texts.size) {
                throw IllegalStateException("Expected ${texts.size} embeddings, but received ${embeddings.size}")
            }
            
            return embeddings.map { it.toFloatArray() }
                .map { if (normalize) VectorUtils.normalize(it) else it }
        } catch (e: Exception) {
            logger.error("Error batch embedding texts with Hugging Face: ${e.message}", e)
            throw EmbeddingException("Failed to generate batch embeddings via Hugging Face API", e)
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
 * Data class for Hugging Face API request
 */
@kotlinx.serialization.Serializable
private data class HuggingFaceRequest(
    val inputs: JsonElement, // Can be a string or a list of strings
    val options: HuggingFaceOptions? = null
)

/**
 * Options for the Hugging Face API request
 */
@kotlinx.serialization.Serializable
private data class HuggingFaceOptions(
    val useCache: Boolean = true,
    val waitForModel: Boolean = true
)
