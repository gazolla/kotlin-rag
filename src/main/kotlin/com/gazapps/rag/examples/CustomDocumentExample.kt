package com.gazapps.rag.examples

import com.gazapps.rag.*
import com.gazapps.rag.core.*
import com.gazapps.rag.core.document.DocumentExtractor
import com.gazapps.rag.core.document.TextPreprocessor
import com.gazapps.rag.core.embedder.OpenAIEmbedder
import com.gazapps.rag.core.llm.OpenAIClient
import com.gazapps.rag.core.vectorstore.InMemoryVectorStore
import kotlinx.coroutines.runBlocking
import java.io.InputStream
import java.util.UUID

/**
 * Example showing how to work with custom document types
 * 
 * This example demonstrates:
 * 1. Creating a custom document class
 * 2. Creating a custom document extractor
 * 3. Indexing custom documents
 * 4. Querying with metadata filters
 */
object CustomDocumentExample {
    
    /**
     * A custom document class representing a product
     */
    data class ProductDocument(
        override val id: String,
        override val content: String,
        override val metadata: Map<String, Any>,
        override val chunks: List<Document>? = null,
        val productId: String,
        val name: String,
        val category: String,
        val price: Double
    ) : Document {
        companion object {
            fun create(
                productId: String,
                name: String,
                description: String,
                category: String,
                price: Double
            ): ProductDocument {
                val id = "product-$productId"
                val content = "Product: $name\nDescription: $description\nCategory: $category\nPrice: $$price"
                val metadata = mapOf(
                    "type" to "product",
                    "product_id" to productId,
                    "name" to name,
                    "category" to category,
                    "price" to price
                )
                
                return ProductDocument(
                    id = id,
                    content = content,
                    metadata = metadata,
                    chunks = null,
                    productId = productId,
                    name = name,
                    category = category,
                    price = price
                )
            }
        }
    }
    
    /**
     * A custom document extractor for CSV files containing product information
     */
    class ProductCsvExtractor : DocumentExtractor {
        override suspend fun extract(input: InputStream, metadata: Map<String, Any>): Document {
            val lines = input.bufferedReader().readLines()
            val header = lines.firstOrNull()?.split(",") ?: emptyList()
            
            if (lines.size <= 1) {
                throw IllegalArgumentException("CSV file has no data rows")
            }
            
            // Find column indices
            val idIdx = header.indexOf("product_id").takeIf { it >= 0 } ?: throw IllegalArgumentException("No product_id column found")
            val nameIdx = header.indexOf("name").takeIf { it >= 0 } ?: throw IllegalArgumentException("No name column found")
            val descIdx = header.indexOf("description").takeIf { it >= 0 } ?: throw IllegalArgumentException("No description column found")
            val categoryIdx = header.indexOf("category").takeIf { it >= 0 } ?: throw IllegalArgumentException("No category column found")
            val priceIdx = header.indexOf("price").takeIf { it >= 0 } ?: throw IllegalArgumentException("No price column found")
            
            // Group content by sections
            val products = mutableListOf<ProductDocument>()
            
            for (i in 1 until lines.size) {
                val values = lines[i].split(",")
                if (values.size < header.size) continue
                
                try {
                    val productId = values[idIdx]
                    val name = values[nameIdx]
                    val description = values[descIdx]
                    val category = values[categoryIdx]
                    val price = values[priceIdx].toDoubleOrNull() ?: 0.0
                    
                    val product = ProductDocument.create(
                        productId = productId,
                        name = name,
                        description = description,
                        category = category,
                        price = price
                    )
                    
                    products.add(product)
                } catch (e: Exception) {
                    // Skip invalid rows
                    println("Error parsing row $i: ${e.message}")
                }
            }
            
            // Create a parent document containing all products
            val id = metadata["id"]?.toString() ?: UUID.randomUUID().toString()
            val content = products.joinToString("\n\n") { it.content }
            
            return SimpleDocument(
                id = id,
                content = content,
                metadata = metadata + mapOf("product_count" to products.size),
                chunks = products
            )
        }
    }
    
    @JvmStatic
    fun main(args: Array<String>) {
        // Replace with your actual API key
        val openAIKey = System.getenv("OPENAI_API_KEY") ?: "your-openai-api-key"
        
        // Create a RAG instance
        val rag = kotlinRag {
            embedder(OpenAIEmbedder(openAIKey))
            vectorStore(InMemoryVectorStore())
            llmClient(OpenAIClient(openAIKey))
            
            config {
                indexing.preprocessText = true
                retrieval.retrievalLimit = 3
            }
        }
        
        runBlocking {
            // Create and index sample product documents
            val products = listOf(
                ProductDocument.create(
                    productId = "P001",
                    name = "Kotlin Programming Language",
                    description = "The Kotlin Programming Language book is the authoritative guide to Kotlin, a modern programming language that makes developers more productive. Written by Kotlin's creators, this book is perfect for developers new to Kotlin.",
                    category = "Books",
                    price = 39.99
                ),
                ProductDocument.create(
                    productId = "P002",
                    name = "IntelliJ IDEA Ultimate",
                    description = "The most intelligent Java IDE with support for Kotlin, Java, Groovy, Scala and other JVM languages, Android, Spring, JavaScript and more.",
                    category = "Software",
                    price = 149.99
                ),
                ProductDocument.create(
                    productId = "P003",
                    name = "Kotlin in Action",
                    description = "Kotlin in Action teaches you to use the Kotlin language for production-quality applications. Written for experienced Java developers, this book covers the features of Kotlin that make it superior to Java.",
                    category = "Books",
                    price = 44.99
                ),
                ProductDocument.create(
                    productId = "P004",
                    name = "Kotlin for Android Developers",
                    description = "A practical book focused on developing Android applications using Kotlin, from the basics to advanced features. Perfect for Android developers looking to transition to Kotlin.",
                    category = "Books",
                    price = 32.99
                )
            )
            
            // Index each product
            products.forEach { product ->
                val success = rag.indexDocument(product)
                println("Indexed ${product.name}: $success")
            }
            
            // Query for all programming books
            val booksQuery = rag.ask(
                "Tell me about programming books",
                filter = mapOf("category" to "Books")
            )
            
            println("\nQuery: Tell me about programming books")
            println("Answer: ${booksQuery.answer}")
            
            // Query for a specific price range using advanced filtering
            val priceFilterOptions = QueryOptions(
                filter = mapOf("type" to "product", "category" to "Books"),
                retrievalLimit = 5
            )
            
            val priceQuery = rag.ask(
                "What books cost less than $40?",
                options = priceFilterOptions
            )
            
            println("\nQuery: What books cost less than $40?")
            println("Answer: ${priceQuery.answer}")
            
            // Print stats
            println("\nDocument Stats:")
            println("Total documents indexed: ${products.size}")
            
            // You could also process a CSV file with the custom extractor
            println("\nTo process a CSV file, you would use:")
            println("val csvFile = File(\"products.csv\")")
            println("val extractor = ProductCsvExtractor()")
            println("val document = extractor.extract(csvFile.inputStream(), mapOf(\"source\" to \"products.csv\"))")
            println("rag.indexDocument(document)")
        }
    }
}
