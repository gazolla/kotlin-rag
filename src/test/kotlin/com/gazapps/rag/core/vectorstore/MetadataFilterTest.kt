package com.gazapps.rag.core.vectorstore

import com.gazapps.rag.core.Document
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class MetadataFilterTest {

    private val testDocument = Document(
        id = "doc1",
        content = "Test content",
        metadata = mapOf(
            "category" to "technology",
            "tags" to listOf("kotlin", "rag"),
            "priority" to 5,
            "published" to true,
            "author" to "John Doe"
        )
    )

    @Test
    fun `eq should match equal values`() {
        val condition = MetadataFilter.eq("category", "technology")
        assertTrue(MetadataFilter.matchesCondition(testDocument, condition))
        
        val nonMatchingCondition = MetadataFilter.eq("category", "science")
        assertFalse(MetadataFilter.matchesCondition(testDocument, nonMatchingCondition))
    }
    
    @Test
    fun `ne should match non-equal values`() {
        val condition = MetadataFilter.ne("category", "science")
        assertTrue(MetadataFilter.matchesCondition(testDocument, condition))
        
        val nonMatchingCondition = MetadataFilter.ne("category", "technology")
        assertFalse(MetadataFilter.matchesCondition(testDocument, nonMatchingCondition))
    }
    
    @Test
    fun `gt should compare numbers correctly`() {
        val condition = MetadataFilter.gt("priority", 3)
        assertTrue(MetadataFilter.matchesCondition(testDocument, condition))
        
        val nonMatchingCondition = MetadataFilter.gt("priority", 7)
        assertFalse(MetadataFilter.matchesCondition(testDocument, nonMatchingCondition))
    }
    
    @Test
    fun `lt should compare numbers correctly`() {
        val condition = MetadataFilter.lt("priority", 7)
        assertTrue(MetadataFilter.matchesCondition(testDocument, condition))
        
        val nonMatchingCondition = MetadataFilter.lt("priority", 3)
        assertFalse(MetadataFilter.matchesCondition(testDocument, nonMatchingCondition))
    }
    
    @Test
    fun `contains should match substrings and list elements`() {
        // String contains
        val stringCondition = MetadataFilter.contains("author", "John")
        assertTrue(MetadataFilter.matchesCondition(testDocument, stringCondition))
        
        // List contains
        val listCondition = MetadataFilter.contains("tags", "kotlin")
        assertTrue(MetadataFilter.matchesCondition(testDocument, listCondition))
        
        // Non-matching
        val nonMatchingCondition = MetadataFilter.contains("tags", "python")
        assertFalse(MetadataFilter.matchesCondition(testDocument, nonMatchingCondition))
    }
    
    @Test
    fun `in should match values in a collection`() {
        val condition = MetadataFilter.`in`("category", listOf("technology", "science"))
        assertTrue(MetadataFilter.matchesCondition(testDocument, condition))
        
        val nonMatchingCondition = MetadataFilter.`in`("category", listOf("science", "math"))
        assertFalse(MetadataFilter.matchesCondition(testDocument, nonMatchingCondition))
    }
    
    @Test
    fun `exists should check if field exists`() {
        val condition = MetadataFilter.exists("category")
        assertTrue(MetadataFilter.matchesCondition(testDocument, condition))
        
        val nonMatchingCondition = MetadataFilter.exists("non_existent_field")
        assertFalse(MetadataFilter.matchesCondition(testDocument, nonMatchingCondition))
    }
    
    @Test
    fun `FilterGroup AND should require all conditions to match`() {
        val group = MetadataFilter.FilterGroup.and(
            MetadataFilter.eq("category", "technology"),
            MetadataFilter.gt("priority", 3)
        )
        
        assertTrue(MetadataFilter.matchesFilterGroup(testDocument, group))
        
        // One condition doesn't match
        val nonMatchingGroup = MetadataFilter.FilterGroup.and(
            MetadataFilter.eq("category", "technology"),
            MetadataFilter.eq("priority", 10)
        )
        
        assertFalse(MetadataFilter.matchesFilterGroup(testDocument, nonMatchingGroup))
    }
    
    @Test
    fun `FilterGroup OR should require at least one condition to match`() {
        val group = MetadataFilter.FilterGroup.or(
            MetadataFilter.eq("category", "science"),
            MetadataFilter.gt("priority", 3)
        )
        
        assertTrue(MetadataFilter.matchesFilterGroup(testDocument, group))
        
        // No conditions match
        val nonMatchingGroup = MetadataFilter.FilterGroup.or(
            MetadataFilter.eq("category", "science"),
            MetadataFilter.eq("priority", 10)
        )
        
        assertFalse(MetadataFilter.matchesFilterGroup(testDocument, nonMatchingGroup))
    }
    
    @Test
    fun `matchesSimpleFilter should match documents correctly`() {
        // Simple equals filter
        val simpleFilter = mapOf("category" to "technology")
        assertTrue(MetadataFilter.matchesSimpleFilter(testDocument, simpleFilter))
        
        // Multiple conditions (AND)
        val multiFilter = mapOf(
            "category" to "technology",
            "published" to true
        )
        assertTrue(MetadataFilter.matchesSimpleFilter(testDocument, multiFilter))
        
        // Non-matching filter
        val nonMatchingFilter = mapOf("category" to "science")
        assertFalse(MetadataFilter.matchesSimpleFilter(testDocument, nonMatchingFilter))
        
        // Empty filter should match all
        val emptyFilter = emptyMap<String, Any>()
        assertTrue(MetadataFilter.matchesSimpleFilter(testDocument, emptyFilter))
        
        // Null filter should match all
        assertTrue(MetadataFilter.matchesSimpleFilter(testDocument, null))
    }
    
    @Test
    fun `fromMap should convert map to filter group`() {
        val map = mapOf(
            "category" to "technology",
            "priority" to 5
        )
        
        val filterGroup = MetadataFilter.fromMap(map)
        
        assertFalse(filterGroup == null)
        filterGroup?.let {
            assertEquals(2, it.conditions.size)
            assertTrue(MetadataFilter.matchesFilterGroup(testDocument, it))
        }
        
        // Null or empty map should return null
        assertEquals(null, MetadataFilter.fromMap(null))
        assertEquals(null, MetadataFilter.fromMap(emptyMap()))
    }
    
    @Test
    fun `nested filter groups should work correctly`() {
        val nestedGroup = MetadataFilter.FilterGroup.or(
            MetadataFilter.FilterGroup.and(
                MetadataFilter.eq("category", "science"),
                MetadataFilter.gt("priority", 3)
            ),
            MetadataFilter.FilterGroup.and(
                MetadataFilter.eq("category", "technology"),
                MetadataFilter.contains("tags", "kotlin")
            )
        )
        
        assertTrue(MetadataFilter.matchesFilterGroup(testDocument, nestedGroup))
    }
}
