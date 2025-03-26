package com.gazapps.rag.core

/**
 * Configuration options for text generation
 *
 * @property maxTokens Maximum number of tokens to generate
 * @property temperature Controls randomness (0.0 = deterministic, 1.0 = creative)
 * @property topP Top-p sampling parameter (0.0-1.0)
 * @property stop List of strings that will stop generation when encountered
 */
data class GenerationOptions(
    val maxTokens: Int = 1024,
    val temperature: Float = 0.7f,
    val topP: Float = 0.95f,
    val stop: List<String> = emptyList()
)

/**
 * Interface for Large Language Model (LLM) clients.
 * 
 * Defines operations for generating text using different LLM providers.
 * Implementations will connect to specific LLM APIs like OpenAI, Anthropic, etc.
 */
interface LLMClient {
    /**
     * Generate text from a prompt.
     *
     * @param prompt The input prompt to generate from
     * @return The generated text
     */
    suspend fun generate(prompt: String): String
    
    /**
     * Generate text from a prompt with specific generation options.
     *
     * @param prompt The input prompt to generate from
     * @param options Configuration options for the generation process
     * @return The generated text
     */
    suspend fun generate(prompt: String, options: GenerationOptions): String
}
