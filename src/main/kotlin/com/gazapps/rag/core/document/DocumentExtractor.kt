package com.gazapps.rag.core.document

import com.gazapps.rag.core.Document
import com.gazapps.rag.core.SimpleDocument
import java.io.File
import java.io.InputStream
import java.nio.charset.StandardCharsets

/**
 * Interface for extracting text content from various document formats
 */
interface DocumentExtractor {
    /**
     * Extract document content from an input stream
     * 
     * @param input Input stream containing the document data
     * @param metadata Optional metadata to include with the document
     * @return Extracted Document
     */
    suspend fun extract(input: InputStream, metadata: Map<String, Any> = emptyMap()): Document
    
    /**
     * Extract document content from a file
     * 
     * @param file File to extract content from
     * @param metadata Optional metadata to include with the document
     * @return Extracted Document
     */
    suspend fun extract(file: File, metadata: Map<String, Any> = emptyMap()): Document {
        return file.inputStream().use { inputStream ->
            val fileMetadata = metadata + mapOf(
                "source" to file.absolutePath,
                "filename" to file.name,
                "last_modified" to file.lastModified(),
                "size" to file.length()
            )
            extract(inputStream, fileMetadata)
        }
    }
}

/**
 * Extractor for plain text files
 */
class TextExtractor : DocumentExtractor {
    override suspend fun extract(input: InputStream, metadata: Map<String, Any>): Document {
        val content = input.readAllBytes().toString(StandardCharsets.UTF_8)
        val id = metadata["filename"]?.toString() ?: "doc-${System.currentTimeMillis()}"
        
        return SimpleDocument(
            id = id,
            content = content,
            metadata = metadata
        )
    }
}

/**
 * Extractor for Markdown files with basic metadata parsing
 */
class MarkdownExtractor : DocumentExtractor {
    override suspend fun extract(input: InputStream, metadata: Map<String, Any>): Document {
        val content = input.readAllBytes().toString(StandardCharsets.UTF_8)
        val id = metadata["filename"]?.toString() ?: "doc-${System.currentTimeMillis()}"
        
        // Extract metadata from frontmatter if present
        val frontMatterMetadata = extractFrontMatter(content)
        val cleanedContent = removeFrontMatter(content)
        
        return SimpleDocument(
            id = id,
            content = cleanedContent,
            metadata = metadata + frontMatterMetadata
        )
    }
    
    /**
     * Extract YAML frontmatter metadata from markdown
     */
    private fun extractFrontMatter(content: String): Map<String, Any> {
        val frontMatterPattern = Regex("^---\\s*\n(.*?)\n---\\s*\n", RegexOption.DOT_MATCHES_ALL)
        val match = frontMatterPattern.find(content) ?: return emptyMap()
        
        val frontMatter = match.groupValues[1]
        val result = mutableMapOf<String, Any>()
        
        // Simple line-by-line parsing of key-value pairs
        frontMatter.lines().forEach { line ->
            val parts = line.split(":", limit = 2)
            if (parts.size == 2) {
                val key = parts[0].trim()
                val value = parts[1].trim()
                if (key.isNotEmpty() && value.isNotEmpty()) {
                    result[key] = value
                }
            }
        }
        
        return result
    }
    
    /**
     * Remove frontmatter from markdown content
     */
    private fun removeFrontMatter(content: String): String {
        val frontMatterPattern = Regex("^---\\s*\n(.*?)\n---\\s*\n", RegexOption.DOT_MATCHES_ALL)
        return frontMatterPattern.replace(content, "")
    }
}

/**
 * HTML extractor that strips HTML tags
 */
class HtmlExtractor : DocumentExtractor {
    override suspend fun extract(input: InputStream, metadata: Map<String, Any>): Document {
        val content = input.readAllBytes().toString(StandardCharsets.UTF_8)
        val id = metadata["filename"]?.toString() ?: "doc-${System.currentTimeMillis()}"
        
        // Extract metadata from HTML meta tags
        val metaTags = extractMetaTags(content)
        
        // Strip HTML tags for content
        val plainText = stripHtmlTags(content)
        
        return SimpleDocument(
            id = id,
            content = plainText,
            metadata = metadata + metaTags
        )
    }
    
    /**
     * Extract metadata from HTML meta tags
     */
    private fun extractMetaTags(html: String): Map<String, Any> {
        val result = mutableMapOf<String, Any>()
        val metaPattern = Regex("<meta\\s+(?:name|property)=\"([^\"]+)\"\\s+content=\"([^\"]+)\"", RegexOption.IGNORE_CASE)
        
        metaPattern.findAll(html).forEach { match ->
            val name = match.groupValues[1]
            val content = match.groupValues[2]
            result[name] = content
        }
        
        // Extract title
        val titlePattern = Regex("<title>([^<]+)</title>", RegexOption.IGNORE_CASE)
        titlePattern.find(html)?.let {
            result["title"] = it.groupValues[1]
        }
        
        return result
    }
    
    /**
     * Strip HTML tags from content
     */
    private fun stripHtmlTags(html: String): String {
        // Basic HTML tag stripping - a full implementation would use a proper HTML parser
        return html
            .replace(Regex("<script[^>]*>.*?</script>", RegexOption.DOT_MATCHES_ALL), "")
            .replace(Regex("<style[^>]*>.*?</style>", RegexOption.DOT_MATCHES_ALL), "")
            .replace(Regex("<!--.*?-->", RegexOption.DOT_MATCHES_ALL), "")
            .replace(Regex("<[^>]*>"), "")
            .replace(Regex("&nbsp;"), " ")
            .replace(Regex("&lt;"), "<")
            .replace(Regex("&gt;"), ">")
            .replace(Regex("&amp;"), "&")
            .replace(Regex("\\s+"), " ")
            .trim()
    }
}

/**
 * Factory methods moved to DocumentExtractorFactory.kt
 */
/*
object DocumentExtractorFactory {
*/