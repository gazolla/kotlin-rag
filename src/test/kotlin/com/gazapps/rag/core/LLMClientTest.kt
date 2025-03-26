package com.gazapps.rag.core

import kotlinx.coroutines.runBlocking
import kotlin.test.Test
import kotlin.test.assertTrue
import kotlin.test.assertContains

class LLMClientTest {
    
    @Test
    fun `MockLLMClient should generate responses based on prompt keywords`() = runBlocking {
        // Setup
        val llmClient = MockLLMClient()
        
        // Act & Assert
        val explanationPrompt = "Please explain how RAG works."
        val explanationResponse = llmClient.generate(explanationPrompt)
        assertTrue(explanationResponse.contains("explanation"), 
                  "Should generate an explanation for 'explain' keyword")
        
        val summaryPrompt = "Summarize the following paragraph: Lorem ipsum dolor sit amet."
        val summaryResponse = llmClient.generate(summaryPrompt)
        assertTrue(summaryResponse.contains("Summary"), 
                  "Should generate a summary for 'summarize' keyword")
        
        val questionPrompt = "What is Retrieval-Augmented Generation?"
        val questionResponse = llmClient.generate(questionPrompt)
        assertTrue(questionResponse.contains("RAG") || questionResponse.contains("Retrieval"), 
                  "Should generate a relevant response for RAG question")
    }
    
    @Test
    fun `MockLLMClient should respect context in prompts`() = runBlocking {
        // Setup
        val llmClient = MockLLMClient()
        
        // Act
        val contextPrompt = """
            Context: RAG combines retrieval systems with generative models to improve responses.
            
            Question: What is RAG?
        """.trimIndent()
        
        val response = llmClient.generate(contextPrompt)
        
        // Assert
        assertContains(response, "RAG combines retrieval systems with generative models")
    }
    
    @Test
    fun `MockLLMClient should handle generation options`() = runBlocking {
        // Setup
        val llmClient = MockLLMClient()
        
        // Act
        // Test temperature effects
        val lowTempOptions = GenerationOptions(temperature = 0.1f)
        val highTempOptions = GenerationOptions(temperature = 0.9f)
        
        val prompt = "What is Kotlin?"
        val lowTempResponse = llmClient.generate(prompt, lowTempOptions)
        val highTempResponse = llmClient.generate(prompt, highTempOptions)
        
        // Assert
        // Higher temperature should lead to longer responses with more variations
        assertTrue(highTempResponse.length >= lowTempResponse.length)
    }
    
    @Test
    fun `MockLLMClient should handle custom templates`() = runBlocking {
        // Setup
        val customResponses = mapOf(
            "custom topic" to "This is a custom response for a special topic.",
            "weather" to "The weather forecast shows {context}."
        )
        val llmClient = MockLLMClient.withCustomBehavior(customResponses)
        
        // Act
        val customPrompt = "Tell me about the custom topic please."
        val customResponse = llmClient.generate(customPrompt)
        
        // Assert
        assertTrue(customResponse.contains("custom response"))
        
        // Test with context
        val weatherPrompt = """
            Context: sunny with a chance of rain later.
            Question: What's the weather forecast?
        """.trimIndent()
        
        val weatherResponse = llmClient.generate(weatherPrompt)
        assertTrue(weatherResponse.contains("sunny with a chance of rain"))
    }
    
    @Test
    fun `MockLLMClient should extract questions correctly`() = runBlocking {
        // Setup
        val llmClient = MockLLMClient()
        
        // Act - question with explicit marker
        val explicitPrompt = """
            Context: Some context here.
            Question: What is the answer?
        """.trimIndent()
        
        val explicitResponse = llmClient.generate(explicitPrompt)
        
        // Act - question without explicit marker
        val implicitPrompt = """
            Context: Some context here.
            
            What is the answer?
        """.trimIndent()
        
        val implicitResponse = llmClient.generate(implicitPrompt)
        
        // Assert - both should extract the question
        assertTrue(explicitResponse.contains("answer") || explicitResponse.contains("context"))
        assertTrue(implicitResponse.contains("answer") || implicitResponse.contains("context"))
    }
}
