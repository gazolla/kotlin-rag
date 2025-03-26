package com.gazapps.rag.core.llm

import com.gazapps.rag.core.ScoredDocument

/**
 * Options for prompt formatting
 */
data class PromptFormatOptions(
    val includeMetadata: Boolean = true,
    val includeSimilarityScores: Boolean = false,
    val formatDocumentNumbers: Boolean = true,
    val maxDocumentsInContext: Int = 5,
    val metadataToInclude: Set<String> = emptySet(),
    val documentSeparator: String = "\n\n",
    val metadataSeparator: String = ": "
)

/**
 * Handles formatting of prompts for LLM clients
 */
class PromptFormatter(
    private val defaultTemplates: Map<PromptTemplate, String> = defaultPromptTemplates()
) {
    /**
     * Format a prompt using a template
     *
     * @param template The template to use
     * @param question The user's question
     * @param documents The retrieved documents
     * @param options Formatting options
     * @return The formatted prompt
     */
    fun format(
        template: PromptTemplate = PromptTemplate.RAG_QA,
        question: String,
        documents: List<ScoredDocument>,
        options: PromptFormatOptions = PromptFormatOptions()
    ): String {
        val templateString = defaultTemplates[template]
            ?: throw IllegalArgumentException("Unknown template: $template")
        
        val context = formatContext(documents, options)
        
        return templateString
            .replace("{context}", context)
            .replace("{question}", question)
    }
    
    /**
     * Format the documents into a context string
     */
    private fun formatContext(
        documents: List<ScoredDocument>,
        options: PromptFormatOptions
    ): String {
        val docsToInclude = documents.take(options.maxDocumentsInContext)
        
        return docsToInclude.mapIndexed { index, scoredDoc ->
            val doc = scoredDoc.document
            val docNum = if (options.formatDocumentNumbers) "${index + 1}" else ""
            
            val metadataSection = if (options.includeMetadata && doc.metadata.isNotEmpty()) {
                formatMetadata(doc.metadata, options)
            } else {
                ""
            }
            
            val similaritySection = if (options.includeSimilarityScores) {
                "[Similarity: ${scoredDoc.score.format(2)}]"
            } else {
                ""
            }
            
            val header = listOf(
                if (docNum.isNotEmpty()) "Document $docNum" else "",
                similaritySection
            ).filter { it.isNotEmpty() }.joinToString(" ")
            
            val headerSection = if (header.isNotEmpty()) "$header\n" else ""
            val metadataPrefix = if (metadataSection.isNotEmpty()) "$metadataSection\n" else ""
            
            "$headerSection$metadataPrefix${doc.content}"
        }.joinToString(options.documentSeparator)
    }
    
    /**
     * Format the metadata for a document
     */
    private fun formatMetadata(
        metadata: Map<String, Any>,
        options: PromptFormatOptions
    ): String {
        val metadataToShow = if (options.metadataToInclude.isNotEmpty()) {
            metadata.filter { (key, _) -> key in options.metadataToInclude }
        } else {
            metadata
        }
        
        if (metadataToShow.isEmpty()) return ""
        
        return metadataToShow.entries.joinToString("\n") { (key, value) ->
            "$key${options.metadataSeparator}$value"
        }
    }
    
    companion object {
        /**
         * Default prompt templates
         */
        fun defaultPromptTemplates(): Map<PromptTemplate, String> = mapOf(
            PromptTemplate.RAG_QA to """
                Based on the following context:
                
                {context}
                
                Answer the question: {question}
                
                If the information needed to answer the question is not in the provided context, say "I don't have enough information to answer that question."
            """.trimIndent(),
            
            PromptTemplate.RAG_SUMMARIZE to """
                Summarize the following documents:
                
                {context}
                
                Question or focus: {question}
            """.trimIndent(),
            
            PromptTemplate.RAG_CONCISE to """
                Context information:
                {context}
                
                Using only the information provided in the context above, provide a brief answer to this question: {question}
                
                Keep your answer concise and focused.
            """.trimIndent(),
            
            PromptTemplate.RAG_DETAILED to """
                I'll provide you with some retrieved documents and a question. Use the documents to provide a detailed answer.
                
                Documents:
                {context}
                
                Question: {question}
                
                Provide a comprehensive answer based only on the information in the documents. If the documents don't contain sufficient information, say so.
            """.trimIndent(),
            
            PromptTemplate.RAG_CHAT to """
                You are an intelligent assistant that references retrieved documents to answer users' questions.
                
                Retrieved information:
                {context}
                
                User question: {question}
                
                Answer conversationally as a helpful assistant, based only on the relevant retrieved information.
            """.trimIndent()
        )
    }
}

/**
 * Predefined prompt templates
 */
enum class PromptTemplate {
    RAG_QA,          // Basic question-answering
    RAG_SUMMARIZE,   // Summarization of documents
    RAG_CONCISE,     // Concise answers
    RAG_DETAILED,    // Detailed, comprehensive answers
    RAG_CHAT         // Conversational style
}

/**
 * Format a float to a string with the specified number of decimal places
 */
private fun Float.format(decimals: Int): String = "%.${decimals}f".format(this)
