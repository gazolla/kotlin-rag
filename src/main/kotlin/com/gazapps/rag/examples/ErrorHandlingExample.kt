package com.gazapps.rag.examples

import com.gazapps.rag.*
import com.gazapps.rag.core.*
import com.gazapps.rag.core.embedder.OpenAIEmbedder
import com.gazapps.rag.core.error.*
import com.gazapps.rag.core.llm.AnthropicClient
import com.gazapps.rag.core.llm.OpenAIClient
import com.gazapps.rag.core.vectorstore.InMemoryVectorStore
import kotlinx.coroutines.runBlocking
import java.io.IOException
import kotlin.random.Random

/**
 * Example showing robust error handling in RAG applications
 * 
 * This example demonstrates:
 * 1. Setting up a RAG instance with error handling
 * 2. Using fallback components
 * 3. Implementing circuit breakers
 * 4. Handling and recovering from different error types
 */
object ErrorHandlingExample {
    
    /**
     * Sample embedder that will fail randomly to demonstrate error handling
     */
    class FlakyEmbedder(private val failureRate: Double = 0.5) : Embedder {
        override suspend fun embed(text: String): FloatArray {
            if (Random.nextDouble() < failureRate) {
                throw IOException("Simulated network failure in embedder")
            }
            
            // Simple mock embedding (don't use in production)
            return FloatArray(128) { Random.nextFloat() }
        }
        
        override suspend fun batchEmbed(texts: List<String>): List<FloatArray> {
            if (Random.nextDouble() < failureRate) {
                throw IOException("Simulated network failure in batch embedder")
            }
            
            // Simple mock embeddings (don't use in production)
            return texts.map { FloatArray(128) { Random.nextFloat() } }
        }
    }
    
    /**
     * Sample LLM client that will fail randomly
     */
    class FlakyLLMClient(private val failureRate: Double = 0.5) : LLMClient {
        override suspend fun generate(prompt: String): String {
            if (Random.nextDouble() < failureRate) {
                throw IOException("Simulated network failure in LLM client")
            }
            
            // Simple mock response
            return "This is a simulated LLM response for the prompt: ${prompt.take(50)}..."
        }
        
        override suspend fun generate(prompt: String, options: GenerationOptions): String {
            if (Random.nextDouble() < failureRate) {
                throw IOException("Simulated network failure in LLM client")
            }
            
            // Simple mock response
            return "This is a simulated LLM response with temperature ${options.temperature} for the prompt: ${prompt.take(50)}..."
        }
    }
    
    @JvmStatic
    fun main(args: Array<String>) {
        // Replace with your actual API keys
        val openAIKey = System.getenv("OPENAI_API_KEY") ?: "your-openai-api-key" 
        val anthropicKey = System.getenv("ANTHROPIC_API_KEY") ?: "your-anthropic-api-key"
        
        // Create primary (flaky) components
        val flakyEmbedder = FlakyEmbedder(failureRate = 0.7) // 70% chance of failure
        val flakyLLM = FlakyLLMClient(failureRate = 0.3)     // 30% chance of failure
        
        // Create reliable fallback components
        val fallbackEmbedder = OpenAIEmbedder(openAIKey) 
        val fallbackLLM = AnthropicClient(anthropicKey)
        
        // Create a circuit breaker for the flaky embedder
        val embedderCircuitBreaker = CircuitBreaker(
            name = "embedder-circuit",
            failureThreshold = 3,       // Open after 3 failures
            resetTimeout = 5000         // Try again after 5 seconds
        )
        
        // Create a circuit breaker for the flaky LLM
        val llmCircuitBreaker = CircuitBreaker(
            name = "llm-circuit",
            failureThreshold = 2,       // Open after 2 failures
            resetTimeout = 3000         // Try again after 3 seconds
        )
        
        // Create a RAG instance with error handling and fallbacks
        val rag = kotlinRag {
            // Primary components (flaky)
            embedder(flakyEmbedder)
            vectorStore(InMemoryVectorStore())
            llmClient(flakyLLM)
            
            // Fallback components (reliable)
            fallbackEmbedder(fallbackEmbedder)
            fallbackLLMClient(fallbackLLM)
            
            // Enable error handling
            withErrorHandling()
            
            // Configure settings
            config {
                // Se seu projeto não tiver estas configurações,
                // podemos remover ou substituir por outras disponíveis
            }
        }
        
        runBlocking {
            // Index sample content (will use fallbacks if primary components fail)
            repeat(5) { i ->
                val content = "This is sample document $i for testing error handling in RAG systems."
                
                try {
                    // Use circuit breaker
                    val success = embedderCircuitBreaker.execute {
                        rag.indexText(content, "doc-$i")
                    }
                    
                    println("Document $i indexed: $success")
                } catch (e: Exception) {
                    println("Failed to index document $i: ${e.message}")
                    
                    // Tentativa de fallback manual
                    try {
                        val success = rag.indexText(content, "doc-$i-fallback")
                        println("Fallback indexing for document $i: $success")
                    } catch (e: Exception) {
                        println("Fallback also failed: ${e.message}")
                    }
                }
            }
            
            // Query with error handling (will use fallbacks if primary components fail)
            repeat(3) { i ->
                val question = "What is document ${i+1} about?"
                
                try {
                    // Use circuit breaker
                    val response = llmCircuitBreaker.execute {
                        rag.ask(question)
                    }
                    
                    println("\nQuestion $i: $question")
                    println("Answer: ${response.answer}")
                } catch (e: Exception) {
                    println("\nFailed to process question $i: ${e.message}")
                    
                    // Handle specific error types
                    when (e) {
                        is RAGException.EmbeddingException -> println("Embedding error occurred")
                        is RAGException.LLMException -> println("LLM error occurred")
                        is RAGException.VectorStoreException -> println("Vector store error occurred")
                        else -> println("Unknown error occurred: ${e.javaClass.simpleName}")
                    }
                    
                    // Tentativa de fallback manual
                    try {
                        val response = rag.ask("Fallback: $question")
                        println("Fallback answer: ${response.answer}")
                    } catch (e: Exception) {
                        println("Fallback query also failed")
                    }
                }
            }
            
            // Demonstração de tratamento de erro básico
            println("\nDemonstrating basic error handling:")
            try {
                if (Random.nextBoolean()) {
                    throw IOException("Simulated IO exception")
                }
                println("Operation completed successfully")
            } catch (e: IOException) {
                println("Handled IO exception: ${e.message}")
            }
            
            // Demonstre retry básico
            println("\nDemonstrating basic retry:")
            var attempts = 0
            var success = false
            
            while (!success && attempts < 3) {
                attempts++
                try {
                    if (attempts < 3) {
                        println("Attempt $attempts: Failing...")
                        throw IOException("Simulated error on attempt $attempts")
                    }
                    
                    println("Attempt $attempts: Succeeding!")
                    success = true
                } catch (e: Exception) {
                    println("Error: ${e.message}")
                    if (attempts >= 3) {
                        println("Max retry attempts reached")
                    }
                }
            }
            
            println("Final result: ${if (success) "Success" else "Failure"} after $attempts attempts")
            
            // Demonstrar logging básico
            println("\nDemonstrating basic logging:")
            println("INFO: This is an info message")
            println("WARN: This is a warning message")
            println("ERROR: This is an error message")
            
            println("\nError handling demonstration complete")
        }
    }
}
