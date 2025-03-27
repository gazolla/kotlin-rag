package com.gazapps.rag.core.llm

import com.gazapps.rag.core.GenerationOptions
import com.gazapps.rag.core.LLMClient
import com.groq.api.client.GroqClientFactory
import com.groq.api.extensions.chatText

/**
 * LLM client implementation for Groq API
 */
class GroqClient(
    private val apiKey: String,
    private val model: String = "llama-3.3-70b-versatile",
    private val temperature: Double = 0.7,
    private val maxTokens: Int = 1024
) : LLMClient {

    /**
     * Generate a response for the given prompt.
     */
    override suspend fun generate(prompt: String): String {
        GroqClientFactory.createClient(apiKey).use { client ->
            // Simple chat completion
            val response = client.chatText(
                model = model,
                userMessage = prompt,
                systemMessage = "You are a helpful assistant."
            )

            return response
        }
    }

    override suspend fun generate(
        prompt: String,
        options: GenerationOptions
    ): String {
        GroqClientFactory.createClient(apiKey).use { client ->
            // Simple chat completion
            val response = client.chatText(
                model = model,
                userMessage = prompt,
                systemMessage = "You are a helpful assistant."
            )

            return response
        }
    }
}
