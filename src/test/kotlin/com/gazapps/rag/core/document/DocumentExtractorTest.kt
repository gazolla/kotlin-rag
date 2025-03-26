package com.gazapps.rag.core.document

import kotlinx.coroutines.runBlocking
import java.io.File
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import kotlin.io.path.createTempFile

class DocumentExtractorTest {

    @Test
    fun `text extractor should extract plain text content`() = runBlocking {
        // Create a temporary text file
        val tempFile = createTempFile("test-", ".txt").toFile()
        tempFile.writeText("This is a test document.\nIt has multiple lines.")
        tempFile.deleteOnExit()
        
        // Create the extractor
        val extractor = TextExtractor()
        
        // Extract content
        val document = extractor.extract(tempFile)
        
        // Verify extraction
        assertEquals(tempFile.name, document.id)
        assertEquals("This is a test document.\nIt has multiple lines.", document.content)
        assertEquals(tempFile.absolutePath, document.metadata["source"])
        assertEquals(tempFile.name, document.metadata["filename"])
        assertTrue(document.metadata.containsKey("last_modified"))
        assertEquals(tempFile.length(), document.metadata["size"])
    }
    
    @Test
    fun `markdown extractor should extract content and frontmatter`() = runBlocking {
        // Create a temporary markdown file with frontmatter
        val tempFile = createTempFile("test-", ".md").toFile()
        tempFile.writeText("""
            ---
            title: Test Document
            author: Test Author
            date: 2023-01-01
            ---
            
            # Test Heading
            
            This is a test markdown document.
            
            ## Section
            
            With multiple sections.
        """.trimIndent())
        tempFile.deleteOnExit()
        
        // Create the extractor
        val extractor = MarkdownExtractor()
        
        // Extract content
        val document = extractor.extract(tempFile)
        
        // Verify extraction
        assertEquals(tempFile.name, document.id)
        assertTrue(document.content.contains("# Test Heading"))
        assertTrue(document.content.contains("With multiple sections"))
        
        // Verify frontmatter was extracted to metadata
        assertEquals("Test Document", document.metadata["title"])
        assertEquals("Test Author", document.metadata["author"])
        assertEquals("2023-01-01", document.metadata["date"])
    }
    
    @Test
    fun `html extractor should extract content without tags`() = runBlocking {
        // Create a temporary HTML file
        val tempFile = createTempFile("test-", ".html").toFile()
        tempFile.writeText("""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Test HTML Document</title>
                <meta name="author" content="Test Author">
            </head>
            <body>
                <h1>Test Heading</h1>
                <p>This is a <strong>test</strong> HTML document.</p>
                <script>
                    // This script content should be removed
                    console.log("Test");
                </script>
            </body>
            </html>
        """.trimIndent())
        tempFile.deleteOnExit()
        
        // Create the extractor
        val extractor = HtmlExtractor()
        
        // Extract content
        val document = extractor.extract(tempFile)
        
        // Verify extraction
        assertEquals(tempFile.name, document.id)
        assertTrue(document.content.contains("Test Heading"))
        assertTrue(document.content.contains("This is a test HTML document"))
        
        // Script content should be removed
        assertTrue(!document.content.contains("console.log"))
        
        // Verify metadata extraction from meta tags
        assertEquals("Test HTML Document", document.metadata["title"])
        assertEquals("Test Author", document.metadata["author"])
    }
    
    @Test
    fun `extractor factory should return appropriate extractor for file type`() {
        // Test text file
        val textExtractor = DocumentExtractorFactory.getExtractorForFile("document.txt")
        assertTrue(textExtractor is TextExtractor)
        
        // Test markdown file
        val mdExtractor = DocumentExtractorFactory.getExtractorForFile("readme.md")
        assertTrue(mdExtractor is MarkdownExtractor)
        
        // Test HTML file
        val htmlExtractor = DocumentExtractorFactory.getExtractorForFile("page.html")
        assertTrue(htmlExtractor is HtmlExtractor)
        
        // Test unknown extension (should default to text)
        val unknownExtractor = DocumentExtractorFactory.getExtractorForFile("data.xyz")
        assertTrue(unknownExtractor is TextExtractor)
    }
}