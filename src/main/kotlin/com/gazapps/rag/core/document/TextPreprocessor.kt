package com.gazapps.rag.core.document

/**
 * Class for preprocessing text before embedding or chunking
 */
class TextPreprocessor(val config: PreprocessingConfig = PreprocessingConfig()) {
    
    /**
     * Preprocess text according to the configuration
     * 
     * @param text Text to preprocess
     * @return Preprocessed text
     */
    fun preprocess(text: String): String {
        var processed = text
        
        // Normalize whitespace
        if (config.normalizeWhitespace) {
            processed = processed.trim().replace(Regex("\\s+"), " ")
        }
        
        // Remove HTML tags if enabled
        if (config.removeHtml) {
            processed = processed.replace(Regex("<[^>]*>"), "")
        }
        
        // Convert to lowercase if enabled
        if (config.lowercase) {
            processed = processed.lowercase()
        }
        
        // Remove punctuation if enabled
        if (config.removePunctuation) {
            processed = processed.replace(Regex("[\\p{P}\\p{S}]"), "")
        }
        
        // Remove diacritics if enabled
        if (config.removeDiacritics) {
            processed = removeDiacritics(processed)
        }
        
        // Replace numbers with tokens if enabled
        if (config.replaceNumbers) {
            processed = replaceNumbers(processed)
        }
        
        // Remove stopwords if enabled
        if (config.removeStopwords) {
            processed = removeStopwords(processed, config.stopwordsList)
        }
        
        // Truncate to max length if specified
        if (config.maxLength != null && processed.length > config.maxLength) {
            processed = processed.substring(0, config.maxLength)
        }
        
        return processed
    }
    
    /**
     * Remove stopwords from text
     */
    private fun removeStopwords(text: String, stopwords: Set<String>): String {
        val words = text.split(Regex("\\s+"))
        return words.filter { it.lowercase() !in stopwords }.joinToString(" ")
    }
    
    /**
     * Remove diacritics (accent marks) from text
     */
    private fun removeDiacritics(text: String): String {
        return text.replace(Regex("[\\p{InCombiningDiacriticalMarks}]"), "")
            .replace('á', 'a').replace('à', 'a').replace('ã', 'a').replace('â', 'a')
            .replace('é', 'e').replace('è', 'e').replace('ê', 'e')
            .replace('í', 'i').replace('ì', 'i').replace('î', 'i')
            .replace('ó', 'o').replace('ò', 'o').replace('õ', 'o').replace('ô', 'o')
            .replace('ú', 'u').replace('ù', 'u').replace('û', 'u')
            .replace('ç', 'c')
            .replace('Á', 'A').replace('À', 'A').replace('Ã', 'A').replace('Â', 'A')
            .replace('É', 'E').replace('È', 'E').replace('Ê', 'E')
            .replace('Í', 'I').replace('Ì', 'I').replace('Î', 'I')
            .replace('Ó', 'O').replace('Ò', 'O').replace('Õ', 'O').replace('Ô', 'O')
            .replace('Ú', 'U').replace('Ù', 'U').replace('Û', 'U')
            .replace('Ç', 'C')
    }

    /**
     * Replace numbers with tokens
     */
    private fun replaceNumbers(text: String): String {
        return text.replace(Regex("\\d+"), "[NUMBER]")
    }
}