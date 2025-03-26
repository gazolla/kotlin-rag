package com.gazapps.rag.core.document

import com.gazapps.rag.core.SimpleDocument
import java.io.InputStream

/**
 * Extrator para arquivos PDF
 */
class PdfExtractor : DocumentExtractor {
    override suspend fun extract(input: InputStream, metadata: Map<String, Any>): com.gazapps.rag.core.Document {
        // Implementação real usaria uma biblioteca para extrair texto de PDF
        val content = "Conteúdo do PDF extraído"
        return SimpleDocument(
            id = metadata["id"]?.toString() ?: System.currentTimeMillis().toString(),
            content = content,
            metadata = metadata + mapOf("format" to "pdf")
        )
    }
}

/**
 * Fábrica para criar extratores de documentos baseados em tipo MIME ou extensão de arquivo
 */
object DocumentExtractorFactory {
    /**
     * Obtém um extrator baseado no tipo MIME
     */
    fun getExtractor(mimeType: String): DocumentExtractor {
        return when {
            mimeType.contains("pdf") -> PdfExtractor()
            mimeType.contains("html") || mimeType.contains("xml") -> HtmlExtractor()
            mimeType.contains("text") -> TextExtractor()
            else -> TextExtractor() // Fallback para extrator de texto simples
        }
    }

    /**
     * Obtém um extrator baseado na extensão do arquivo
     */
    fun getExtractorForFile(fileName: String): DocumentExtractor {
        val extension = fileName.substringAfterLast('.', "").lowercase()
        return getExtractorForExtension(extension)
    }

    fun getExtractorForExtension(extension: String): DocumentExtractor {
        return when (extension.lowercase()) {
            "pdf" -> PdfExtractor()
            "html", "htm", "xml" -> HtmlExtractor()
            "txt", "md", "csv", "json" -> TextExtractor()
            "markdown" -> MarkdownExtractor()
            else -> TextExtractor() // Fallback para extrator de texto simples
        }
    }
}
