package com.gazapps.rag.core.llm

import com.gazapps.rag.core.GenerationOptions
import com.gazapps.rag.core.LLMClient
import com.gazolla.groq.GroqApi
import com.gazolla.groq.ChatCompletionRequest
import com.gazolla.groq.Message
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * LLM client implementation for Groq API
 */
class GroqClient(
    private val apiKey: String,
    private val model: String = "llama3-70b-8192",
    private val temperature: Double = 0.7,
    private val maxTokens: Int = 1024
) : LLMClient {

    private val groqApi = GroqApi(apiKey)

    /**
     * Generate a response for the given prompt.
     */
    override suspend fun generate(prompt: String): String {
        return withContext(Dispatchers.IO) {
            val request = ChatCompletionRequest(
                messages = listOf(Message(role = "user", content = prompt)),
                model = model,
                temperature = temperature,
                maxTokens = maxTokens
            )
            
            val response = groqApi.chatCompletion(request)
            response.choices.firstOrNull()?.message?.content ?: "No response generated"
        }
    }

    /**
     * Generate a response with advanced options.
     */
    override suspend fun generate(prompt: String, options: GenerationOptions): String {
        return withContext(Dispatchers.IO) {
            val request = ChatCompletionRequest(
                messages = listOf(Message(role = "user", content = prompt)),
                model = model,
                temperature = options.temperature.toDouble(),
                maxTokens = options.maxTokens,
                topP = options.topP.toDouble(),
                stop = options.stop.takeIf { it.isNotEmpty() }
            )
            
            val response = groqApi.chatCompletion(request)
            response.choices.firstOrNull()?.message?.content ?: "No response generated"
        }
    }
}
