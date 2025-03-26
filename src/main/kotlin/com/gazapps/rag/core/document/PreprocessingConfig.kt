package com.gazapps.rag.core.document

/**
 * Configuração para pré-processamento de texto.
 * Incluindo opções avançadas de processamento de texto.
 */
data class PreprocessingConfig(
    /**
     * Se deve normalizar espaços em branco.
     */
    val normalizeWhitespace: Boolean = true,
    
    /**
     * Se deve remover tags HTML.
     */
    val removeHtml: Boolean = true,
    
    /**
     * Se deve converter o texto para lowercase.
     */
    val lowercase: Boolean = false,
    
    /**
     * Se deve remover palavras comuns (stopwords).
     */
    val removeStopwords: Boolean = false,
    
    /**
     * Se deve remover pontuação.
     */
    val removePunctuation: Boolean = false,
    
    /**
     * Lista de palavras comuns (stopwords) a serem removidas.
     */
    val stopwordsList: Set<String> = DEFAULT_STOPWORDS,
    
    /**
     * Se deve remover acentos diacríticos.
     */
    val removeDiacritics: Boolean = false,
    
    /**
     * Se deve substituir números com tokens.
     */
    val replaceNumbers: Boolean = false,
    
    /**
     * Comprimento máximo do texto processado (opcional).
     */
    val maxLength: Int? = null
) {
    companion object {
        /**
         * Lista padrão de stopwords em português.
         */
        val DEFAULT_STOPWORDS = setOf(
            "a", "e", "o", "as", "os", "um", "uma", "uns", "umas", "de", "da", "do",
            "das", "dos", "em", "na", "no", "nas", "nos", "por", "para", "com",
            "ao", "à", "às", "pelo", "pela", "pelos", "pelas", "que", "se", "não"
        )
    }
    
    /**
     * Verifica se a configuração tem remoção de diacríticos ativada.
     */
    fun hasRemoveDiacritics(): Boolean = removeDiacritics
    
    /**
     * Verifica se a configuração tem substituição de números ativada.
     */
    fun hasReplaceNumbers(): Boolean = replaceNumbers
    
    /**
     * Verifica se a configuração tem comprimento máximo definido.
     */
    fun hasMaxLength(): Boolean = maxLength != null
    
    /**
     * Retorna o comprimento máximo ou um valor padrão se não definido.
     */
    fun getMaxLength(): Int = maxLength ?: Int.MAX_VALUE
}

