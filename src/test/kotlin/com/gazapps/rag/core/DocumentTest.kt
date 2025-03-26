package com.gazapps.rag.core

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull

class DocumentTest {
    
    @Test
    fun `SimpleDocument should correctly store and retrieve properties`() {
        // Setup
        val id = "doc123"
        val content = "This is test content"
        val metadata = mapOf("source" to "test", "author" to "tester")
        
        // Act
        val document = SimpleDocument(id, content, metadata)
        
        // Assert
        assertEquals(id, document.id)
        assertEquals(content, document.content)
        assertEquals(metadata, document.metadata)
        assertNull(document.chunks)
    }
    
    @Test
    fun `SimpleDocument with chunks should correctly store and retrieve chunks`() {
        // Setup
        val childDoc1 = SimpleDocument("child1", "Child content 1")
        val childDoc2 = SimpleDocument("child2", "Child content 2")
        val chunks = listOf(childDoc1, childDoc2)
        
        // Act
        val parentDoc = SimpleDocument("parent", "Parent content", chunks = chunks)
        
        // Assert
        assertEquals(2, parentDoc.chunks?.size)
        assertEquals(childDoc1, parentDoc.chunks?.get(0))
        assertEquals(childDoc2, parentDoc.chunks?.get(1))
    }
    
    @Test
    fun `withMetadata extension function should add metadata correctly`() {
        // Setup
        val document = SimpleDocument("doc1", "Test content", mapOf("key1" to "value1"))
        
        // Act
        val updatedDoc = document.withMetadata("key2", "value2")
        
        // Assert
        assertEquals(document.id, updatedDoc.id)
        assertEquals(document.content, updatedDoc.content)
        assertEquals(2, updatedDoc.metadata.size)
        assertEquals("value1", updatedDoc.metadata["key1"])
        assertEquals("value2", updatedDoc.metadata["key2"])
    }
    
    @Test
    fun `withMetadata extension function should handle map of metadata`() {
        // Setup
        val document = SimpleDocument("doc1", "Test content", mapOf("key1" to "value1"))
        val additionalMetadata = mapOf("key2" to "value2", "key3" to 123)
        
        // Act
        val updatedDoc = document.withMetadata(additionalMetadata)
        
        // Assert
        assertEquals(3, updatedDoc.metadata.size)
        assertEquals("value1", updatedDoc.metadata["key1"])
        assertEquals("value2", updatedDoc.metadata["key2"])
        assertEquals(123, updatedDoc.metadata["key3"])
    }
    
    @Test
    fun `toDocument extension function should create document from string`() {
        // Setup
        val text = "This is a test string"
        val metadata = mapOf("source" to "test")
        
        // Act
        val document = text.toDocument(id = "doc1", metadata = metadata)
        
        // Assert
        assertEquals("doc1", document.id)
        assertEquals(text, document.content)
        assertEquals(metadata, document.metadata)
    }
    
    @Test
    fun `toDocument should generate ID when not provided`() {
        // Setup
        val text = "This is a test string"
        
        // Act
        val document = text.toDocument()
        
        // Assert
        assertTrue(document.id.startsWith("doc-"))
        assertEquals(text, document.content)
    }
    
    private fun assertTrue(condition: Boolean) {
        assert(condition)
    }
}
