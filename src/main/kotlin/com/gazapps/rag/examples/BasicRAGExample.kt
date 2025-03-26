package com.gazapps.rag.examples

import com.gazapps.rag.*
import com.gazapps.rag.core.*
import com.gazapps.rag.core.embedder.OpenAIEmbedder
import com.gazapps.rag.core.llm.AnthropicClient
import com.gazapps.rag.core.vectorstore.InMemoryVectorStore
import kotlinx.coroutines.runBlocking
import java.io.File

/**
 * Basic example showing how to set up and use a RAG system
 * 
 * This example demonstrates:
 * 1. Creating a RAG instance using the DSL
 * 2. Indexing documents from text and files
 * 3. Asking questions and getting responses
 */
object BasicRAGExample {
    @JvmStatic
    fun main(args: Array<String>) {
        // Replace with your actual API keys
        val openAIKey = System.getenv("OPENAI_API_KEY") ?: "your-openai-api-key" 
        val anthropicKey = System.getenv("ANTHROPIC_API_KEY") ?: "your-anthropic-api-key"
        
        // Create a RAG instance
        val rag = kotlinRag {
            // Configure components
            embedder(OpenAIEmbedder(openAIKey))
            vectorStore(InMemoryVectorStore())
            llmClient(AnthropicClient(anthropicKey))
            
            // Configure RAG settings
            config {
                // Customize chunk size
                indexing.chunkSize = 500
                indexing.chunkOverlap = 50
                
                // Customize retrieval settings
                retrieval.retrievalLimit = 3
                retrieval.reranking = true
                
                // Customize prompt template
                generation.promptTemplate = """
                    Based on the following context, please answer the question.
                    
                    Context:
                    {context}
                    
                    Question:
                    {question}
                    
                    Answer:
                """.trimIndent()
            }
        }
        
        runBlocking {
            // Index some content
            val indexed = rag.indexText(
                content = "Kotlin is a modern programming language that makes developers more productive. " +
                          "Kotlin is concise, safe, interoperable with Java, and supports multiple platforms.",
                metadata = mapOf("source" to "kotlin-intro", "topic" to "programming")
            )
            
            println("Indexing successful: $indexed")
            
            // Ask a question
            val response = rag.ask("What is Kotlin?")
            
            println("Question: What is Kotlin?")
            println("Answer: ${response.answer}")
            println("Retrieved documents: ${response.documents.size}")
            println("Processing time: ${response.processingTimeMs}ms")
            
            // You can also ask with filtering by metadata
            val filteredResponse = rag.ask(
                question = "What platforms does Kotlin support?",
                filter = mapOf("topic" to "programming")
            )
            
            println("\nQuestion: What platforms does Kotlin support?")
            println("Answer: ${filteredResponse.answer}")
        }
    }
}
