package com.gazapps.rag.core.llm

import com.gazapps.rag.core.GenerationOptions
import com.gazapps.rag.core.LLMClient

/**
 * Uma implementação simples de LLMClient para propósitos de teste.
 * Gera respostas determinísticas baseadas na pergunta.
 */
class MockLLMClient(
    private val defaultPrefix: String = "Resposta simulada para: ",
    private val respectContexts: Boolean = true
) : LLMClient {
    
    /**
     * Gera uma resposta simulada para o prompt fornecido.
     * 
     * @param prompt O prompt para gerar a resposta
     * @return Uma resposta simulada
     */
    override suspend fun generate(prompt: String): String {
        // Simplificado para testes
        if (!respectContexts || !prompt.contains("contexto", ignoreCase = true)) {
            return "$defaultPrefix ${prompt.take(50)}..."
        }
        
        // Se estamos respeitando contextos, extrair a pergunta e o contexto
        val questionRegex = Regex("(?:pergunta|question):\\s*([^\\n]+)", RegexOption.IGNORE_CASE)
        val question = questionRegex.find(prompt)?.groupValues?.get(1) ?: prompt.takeLast(50)
        
        return gerarRespostaSensivel(question, prompt)
    }
    
    /**
     * Gera uma resposta simulada para o prompt fornecido com opções adicionais.
     * 
     * @param prompt O prompt para gerar a resposta
     * @param options Opções de geração
     * @return Uma resposta simulada
     */
    override suspend fun generate(prompt: String, options: GenerationOptions): String {
        val baseResponse = generate(prompt)
        
        // Adaptar a resposta conforme as opções
        val responseLength = if (options.maxTokens < 100) "curta" else "detalhada"
        val creativity = if (options.temperature < 0.5f) "factual" else "criativa"
        
        return "[$responseLength e $creativity] $baseResponse"
    }
    
    /**
     * Gera uma resposta que parece usar informações do contexto.
     */
    private fun gerarRespostaSensivel(question: String, fullPrompt: String): String {
        // Extrair alguns fragmentos do contexto para simular uma resposta baseada nele
        val contextFragments = fullPrompt
            .split('\n')
            .filter { it.length > 30 && !it.contains("pergunta", ignoreCase = true) }
            .shuffled()
            .take(2)
            .joinToString(" ") { it.trim() }
        
        val shortenedContext = if (contextFragments.length > 100) {
            contextFragments.substring(0, 100) + "..."
        } else {
            contextFragments
        }
        
        return "Com base no contexto fornecido, posso responder que $shortenedContext " +
               "Esta informação responde à sua pergunta: '$question'"
    }
}
