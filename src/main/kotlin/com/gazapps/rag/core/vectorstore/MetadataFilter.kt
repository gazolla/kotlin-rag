package com.gazapps.rag.core.vectorstore

import kotlin.ranges.ClosedRange

import com.gazapps.rag.core.Document
import java.time.Instant

/**
 * Advanced metadata filtering utility for vector stores
 */
object MetadataFilter {
    /**
     * Operator for comparison operations
     */
    enum class ComparisonOperator {
        EQUALS,         // Equal to
        NOT_EQUALS,     // Not equal to
        GREATER_THAN,   // Greater than
        LESS_THAN,      // Less than
        GREATER_EQUALS, // Greater than or equal to
        LESS_EQUALS,    // Less than or equal to
        CONTAINS,       // Contains substring/element
        IN,             // Value is in a list
        NOT_IN,         // Value is not in a list
        BETWEEN,        // Value is between min and max
        MATCHES,        // Value matches a regex pattern
        EXISTS          // Field exists
    }

    /**
     * Filter condition for a single field
     */
    data class Condition(
        val field: String,
        val operator: ComparisonOperator,
        val value: Any?
    )

    /**
     * Group of conditions with a logical operator
     */
    data class FilterGroup(
        val conditions: List<Any>, // Can be Condition or nested FilterGroup
        val operator: LogicalOperator = LogicalOperator.AND
    ) {
        enum class LogicalOperator {
            AND, OR
        }

        companion object {
            /**
             * Create an AND filter group
             */
            fun and(vararg conditions: Any): FilterGroup {
                return FilterGroup(conditions.toList(), LogicalOperator.AND)
            }

            /**
             * Create an OR filter group
             */
            fun or(vararg conditions: Any): FilterGroup {
                return FilterGroup(conditions.toList(), LogicalOperator.OR)
            }
        }
    }

    /**
     * Create a simple equals condition
     */
    fun eq(field: String, value: Any?): Condition {
        return Condition(field, ComparisonOperator.EQUALS, value)
    }

    /**
     * Create a not equals condition
     */
    fun ne(field: String, value: Any?): Condition {
        return Condition(field, ComparisonOperator.NOT_EQUALS, value)
    }

    /**
     * Create a greater than condition
     */
    fun gt(field: String, value: Number): Condition {
        return Condition(field, ComparisonOperator.GREATER_THAN, value)
    }

    /**
     * Create a less than condition
     */
    fun lt(field: String, value: Number): Condition {
        return Condition(field, ComparisonOperator.LESS_THAN, value)
    }

    /**
     * Create a greater than or equal condition
     */
    fun gte(field: String, value: Number): Condition {
        return Condition(field, ComparisonOperator.GREATER_EQUALS, value)
    }

    /**
     * Create a less than or equal condition
     */
    fun lte(field: String, value: Number): Condition {
        return Condition(field, ComparisonOperator.LESS_EQUALS, value)
    }

    /**
     * Create a contains condition
     */
    fun contains(field: String, value: Any): Condition {
        return Condition(field, ComparisonOperator.CONTAINS, value)
    }

    /**
     * Create an in condition
     */
    fun `in`(field: String, values: Collection<Any>): Condition {
        return Condition(field, ComparisonOperator.IN, values)
    }

    /**
     * Create a not in condition
     */
    fun notIn(field: String, values: Collection<Any>): Condition {
        return Condition(field, ComparisonOperator.NOT_IN, values)
    }

    /**
     * Create a between condition
     */
    fun between(field: String, min: Number, max: Number): Condition {
        return Condition(field, ComparisonOperator.BETWEEN, listOf(min, max))
    }

    /**
     * Create a matches condition (regex)
     */
    fun matches(field: String, pattern: String): Condition {
        return Condition(field, ComparisonOperator.MATCHES, pattern)
    }

    /**
     * Create an exists condition
     */
    fun exists(field: String): Condition {
        return Condition(field, ComparisonOperator.EXISTS, null)
    }

    /**
     * Check if a document matches a condition
     */
    fun matchesCondition(document: Document, condition: Condition): Boolean {
        val field = condition.field
        val value = document.metadata[field]

        return when (condition.operator) {
            ComparisonOperator.EQUALS -> value == condition.value
            ComparisonOperator.NOT_EQUALS -> value != condition.value
            ComparisonOperator.GREATER_THAN -> compareValues(value, condition.value) > 0
            ComparisonOperator.LESS_THAN -> compareValues(value, condition.value) < 0
            ComparisonOperator.GREATER_EQUALS -> compareValues(value, condition.value) >= 0
            ComparisonOperator.LESS_EQUALS -> compareValues(value, condition.value) <= 0
            ComparisonOperator.CONTAINS -> containsValue(value, condition.value)
            ComparisonOperator.IN -> {
                val collection = condition.value as? Collection<*> ?: return false
                collection.contains(value)
            }
            ComparisonOperator.NOT_IN -> {
                val collection = condition.value as? Collection<*> ?: return false
                !collection.contains(value)
            }
            ComparisonOperator.BETWEEN -> {
                val range = condition.value as? List<*> ?: return false
                val min = range.getOrNull(0) as? Number ?: return false
                val max = range.getOrNull(1) as? Number ?: return false
                val numberValue = value as? Number ?: return false
                numberValue.toDouble() in min.toDouble()..max.toDouble()
            }
            ComparisonOperator.MATCHES -> {
                val pattern = condition.value as? String ?: return false
                val stringValue = value?.toString() ?: return false
                Regex(pattern).matches(stringValue)
            }
            ComparisonOperator.EXISTS -> document.metadata.containsKey(field)
        }
    }

    /**
     * Check if a document matches a filter group
     */
    fun matchesFilterGroup(document: Document, group: FilterGroup): Boolean {
        val results = group.conditions.map { condition ->
            when (condition) {
                is Condition -> matchesCondition(document, condition)
                is FilterGroup -> matchesFilterGroup(document, condition)
                else -> false
            }
        }

        return when (group.operator) {
            FilterGroup.LogicalOperator.AND -> results.all { it }
            FilterGroup.LogicalOperator.OR -> results.any { it }
        }
    }

    /**
     * Filter a list of documents using a filter group
     */
    fun applyFilter(documents: List<Document>, filter: FilterGroup): List<Document> {
        return documents.filter { matchesFilterGroup(it, filter) }
    }

    /**
     * Simple matcher for basic Map<String, Any> filters
     */
    fun matchesSimpleFilter(document: Document, filter: Map<String, Any>?): Boolean {
        if (filter.isNullOrEmpty()) return true

        return filter.all { (key, value) ->
            when (value) {
                is Collection<*> -> document.metadata[key]?.let { value.contains(it) } ?: false
                is ClosedRange<*> -> {
                    val metaValue = document.metadata[key] as? Number ?: return@all false
                    val min = (value.start as? Number)?.toDouble() ?: return@all false
                    val max = (value.endInclusive as? Number)?.toDouble() ?: return@all false
                    metaValue.toDouble() in min..max
                }
                is Regex -> (document.metadata[key] as? String)?.matches(value) ?: false
                else -> document.metadata[key] == value
            }
        }
    }

    /**
     * Helper function to compare values safely
     */
    @Suppress("UNCHECKED_CAST")
    private fun compareValues(a: Any?, b: Any?): Int {
        if (a == null || b == null) return 0

        return when {
            a is Number && b is Number -> {
                a.toDouble().compareTo(b.toDouble())
            }
            a is String && b is String -> {
                a.compareTo(b)
            }
            a is Instant && b is Instant -> {
                a.compareTo(b)
            }
            a is Comparable<*> && b::class == a::class -> {
                (a as Comparable<Any>).compareTo(b)
            }
            else -> 0
        }
    }

    /**
     * Helper function to check if a value contains another value
     */
    private fun containsValue(container: Any?, value: Any?): Boolean {
        if (container == null || value == null) return false

        return when (container) {
            is String -> {
                val valueStr = value.toString()
                container.contains(valueStr)
            }
            is Collection<*> -> container.contains(value)
            is Array<*> -> container.contains(value)
            is Map<*, *> -> container.containsValue(value) || container.containsKey(value)
            else -> false
        }
    }

    /**
     * Create a filter group from a simple Map<String, Any>
     */
    fun fromMap(filter: Map<String, Any>?): FilterGroup? {
        if (filter.isNullOrEmpty()) return null

        val conditions = filter.map { (key, value) -> eq(key, value) }
        return FilterGroup(conditions, FilterGroup.LogicalOperator.AND)
    }
}
