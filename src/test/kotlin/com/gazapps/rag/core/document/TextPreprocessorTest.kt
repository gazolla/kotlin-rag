package com.gazapps.rag.core.document

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotEquals
import kotlin.test.assertTrue

class TextPreprocessorTest {

    private val sampleText = """
        This is a  sample   text with irregular spacing.
        
        It contains multiple paragraphs and some HTML <b>tags</b>.
        
        It also has numbers like 12345 and punctuation marks!
        
        Some accented characters: café, résumé.
    """.trimIndent()
    
    @Test
    fun `should normalize whitespace`() {
        val preprocessor = TextPreprocessor(PreprocessingConfig(
            normalizeWhitespace = true
        ))
        
        val processed = preprocessor.preprocess(sampleText)
        
        // Should not contain double spaces
        assertTrue(!processed.contains("  "))
        
        // Should have converted newlines to spaces
        assertTrue(!processed.contains("\n\n"))
    }
    
    @Test
    fun `should remove HTML tags`() {
        val preprocessor = TextPreprocessor(PreprocessingConfig(
            removeHtml = true
        ))
        
        val processed = preprocessor.preprocess(sampleText)
        
        // HTML tags should be removed
        assertTrue(!processed.contains("<b>"))
        assertTrue(!processed.contains("</b>"))
        assertTrue(processed.contains("tags"))
    }
    
    @Test
    fun `should convert to lowercase`() {
        val preprocessor = TextPreprocessor(PreprocessingConfig(
            lowercase = true
        ))
        
        val processed = preprocessor.preprocess(sampleText)
        
        // All text should be lowercase
        assertEquals(processed.lowercase(), processed)
        assertTrue(processed.contains("this is"))
    }
    
    @Test
    fun `should remove punctuation`() {
        val preprocessor = TextPreprocessor(PreprocessingConfig(
            removePunctuation = true
        ))
        
        val processed = preprocessor.preprocess(sampleText)
        
        // Punctuation should be removed
        assertTrue(!processed.contains("!"))
        assertTrue(!processed.contains("."))
    }
    
    @Test
    fun `should remove diacritics`() {
        val preprocessor = TextPreprocessor(PreprocessingConfig(
            removeDiacritics = true
        ))
        
        val processed = preprocessor.preprocess(sampleText)
        
        // Accented characters should be normalized
        assertTrue(!processed.contains("é"))
        assertTrue(processed.contains("cafe"))
        assertTrue(processed.contains("resume"))
    }
    
    @Test
    fun `should replace numbers`() {
        val preprocessor = TextPreprocessor(PreprocessingConfig(
            replaceNumbers = true
        ))
        
        val processed = preprocessor.preprocess(sampleText)
        
        // Numbers should be replaced with token
        assertTrue(!processed.contains("12345"))
        assertTrue(processed.contains("[NUMBER]"))
    }
    
    @Test
    fun `should remove stopwords`() {
        val preprocessor = TextPreprocessor(PreprocessingConfig(
            removeStopwords = true
        ))
        
        val processed = preprocessor.preprocess(sampleText)
        
        // Common stopwords should be removed
        assertTrue(!processed.contains(" is "))
        assertTrue(!processed.contains(" a "))
        assertTrue(!processed.contains(" with "))
        
        // Content words should remain
        assertTrue(processed.contains("sample"))
        assertTrue(processed.contains("text"))
    }
    
    @Test
    fun `should combine multiple preprocessing steps`() {
        val preprocessor = TextPreprocessor(PreprocessingConfig(
            normalizeWhitespace = true,
            removeHtml = true,
            lowercase = true,
            removePunctuation = true
        ))
        
        val processed = preprocessor.preprocess(sampleText)
        
        // Should apply all transformations
        assertEquals(processed.lowercase(), processed)
        assertTrue(!processed.contains("<b>"))
        assertTrue(!processed.contains("  "))
        assertTrue(!processed.contains("!"))
    }
    
    @Test
    fun `should truncate to max length`() {
        val maxLength = 20
        val preprocessor = TextPreprocessor(PreprocessingConfig(
            maxLength = maxLength
        ))
        
        val processed = preprocessor.preprocess(sampleText)
        
        // Text should be truncated to maxLength
        assertEquals(maxLength, processed.length)
    }
}