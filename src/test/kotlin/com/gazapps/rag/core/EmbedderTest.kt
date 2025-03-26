package com.gazapps.rag.core

import kotlinx.coroutines.runBlocking
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class EmbedderTest {
    
    @Test
    fun `DummyEmbedder should generate embeddings with correct dimensions`() = runBlocking {
        // Setup
        val embedder = DummyEmbedder(dimensions = 512)
        val text = "This is a test text"
        
        // Act
        val embedding = embedder.embed(text)
        
        // Assert
        assertEquals(512, embedding.size)
    }
    
    @Test
    fun `DummyEmbedder should generate normalized embeddings`() = runBlocking {
        // Setup
        val embedder = DummyEmbedder()
        val text = "Test normalization"
        
        // Act
        val embedding = embedder.embed(text)
        
        // Assert
        val magnitude = embedding.fold(0f) { acc, value -> acc + value * value }
        assertTrue(kotlin.math.abs(1.0f - kotlin.math.sqrt(magnitude)) < 0.00001f, 
                  "Embedding should be normalized with magnitude of 1.0")
    }
    
    @Test
    fun `DummyEmbedder should generate consistent embeddings for same text when deterministic`() = runBlocking {
        // Setup
        val embedder = DummyEmbedder(deterministic = true)
        val text = "Consistent embedding test"
        
        // Act
        val embedding1 = embedder.embed(text)
        val embedding2 = embedder.embed(text)
        
        // Assert
        for (i in embedding1.indices) {
            assertEquals(embedding1[i], embedding2[i], "Embeddings should be consistent for the same text")
        }
    }
    
    @Test
    fun `DummyEmbedder batchEmbed should return correct number of embeddings`() = runBlocking {
        // Setup
        val embedder = DummyEmbedder()
        val texts = listOf("Text 1", "Text 2", "Text 3")
        
        // Act
        val embeddings = embedder.batchEmbed(texts)
        
        // Assert
        assertEquals(3, embeddings.size)
        assertEquals(embedder.embed(texts[0]).toList(), embeddings[0].toList())
        assertEquals(embedder.embed(texts[1]).toList(), embeddings[1].toList())
        assertEquals(embedder.embed(texts[2]).toList(), embeddings[2].toList())
    }
    
    @Test
    fun `SemanticEmbedder should make similar concepts have similar embeddings`() = runBlocking {
        // Setup
        val embedder = DummyEmbedder.createSemanticEmbedder()
        
        // Act
        val kotlinEmbedding = embedder.embed("kotlin programming")
        val javaEmbedding = embedder.embed("java programming")
        val aiEmbedding = embedder.embed("artificial intelligence")
        
        // Compare similarities
        val kotlinJavaSimilarity = VectorUtils.cosineSimilarity(kotlinEmbedding, javaEmbedding)
        val kotlinAiSimilarity = VectorUtils.cosineSimilarity(kotlinEmbedding, aiEmbedding)
        
        // Assert
        assertTrue(kotlinJavaSimilarity > kotlinAiSimilarity, 
            "Programming languages should be more similar to each other than to AI concepts")
    }
}
