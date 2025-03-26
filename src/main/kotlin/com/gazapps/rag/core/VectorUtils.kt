package com.gazapps.rag.core

import kotlin.math.sqrt

/**
 * Utilitários para operações com vetores.
 */
object VectorUtils {
    /**
     * Calcula a similaridade de cosseno entre dois vetores.
     *
     * @param a Primeiro vetor
     * @param b Segundo vetor
     * @return Valor de similaridade entre 0 e 1
     */
    fun cosineSimilarity(a: FloatArray, b: FloatArray): Float {
        require(a.size == b.size) { "Vectors must have same dimensions" }
        
        var dotProduct = 0f
        var normA = 0f
        var normB = 0f
        
        for (i in a.indices) {
            dotProduct += a[i] * b[i]
            normA += a[i] * a[i]
            normB += b[i] * b[i]
        }
        
        return if (normA > 0f && normB > 0f) {
            dotProduct / (sqrt(normA) * sqrt(normB))
        } else {
            0f
        }
    }
    
    /**
     * Calcula a distância euclidiana entre dois vetores.
     *
     * @param a Primeiro vetor
     * @param b Segundo vetor
     * @return Valor da distância euclidiana
     */
    fun euclideanDistance(a: FloatArray, b: FloatArray): Float {
        require(a.size == b.size) { "Vectors must have same dimensions" }
        
        var sum = 0f
        for (i in a.indices) {
            val diff = a[i] - b[i]
            sum += diff * diff
        }
        
        return sqrt(sum)
    }
    
    /**
     * Calcula o produto escalar (dot product) entre dois vetores.
     *
     * @param a Primeiro vetor
     * @param b Segundo vetor
     * @return Valor do produto escalar
     */
    fun dotProduct(a: FloatArray, b: FloatArray): Float {
        require(a.size == b.size) { "Vectors must have same dimensions" }
        
        var product = 0f
        for (i in a.indices) {
            product += a[i] * b[i]
        }
        
        return product
    }
    
    /**
     * Normaliza um vetor para ter magnitude 1 (vetor unitário).
     * Modifica o vetor original.
     *
     * @param vector Vetor a ser normalizado in-place
     * @return O vetor original, normalizado
     */
    fun normalize(vector: FloatArray): FloatArray {
        var norm = 0f
        for (value in vector) {
            norm += value * value
        }
        norm = sqrt(norm)
        
        if (norm > 0f) {
            for (i in vector.indices) {
                vector[i] = vector[i] / norm
            }
        }
        
        return vector
    }
    
    /**
     * Cria e retorna uma cópia normalizada do vetor, sem modificar o original.
     *
     * @param vector Vetor a ser normalizado
     * @return Uma nova instância do vetor normalizado
     */
    fun normalized(vector: FloatArray): FloatArray {
        var norm = 0f
        for (value in vector) {
            norm += value * value
        }
        norm = sqrt(norm)
        
        return if (norm > 0f) {
            FloatArray(vector.size) { i -> vector[i] / norm }
        } else {
            vector.clone()
        }
    }
    
    /**
     * Calcula a média de um conjunto de vetores.
     *
     * @param vectors Lista de vetores
     * @return Vetor médio
     */
    fun average(vectors: List<FloatArray>): FloatArray {
        if (vectors.isEmpty()) {
            return floatArrayOf()
        }
        
        val dimensions = vectors.first().size
        vectors.forEach { 
            require(it.size == dimensions) { "All vectors must have the same dimensions" }
        }
        
        // Inicializar um array de zeros
        val result = FloatArray(dimensions) { 0f }
        
        // Somar os vetores
        for (vector in vectors) {
            for (i in result.indices) {
                result[i] = result[i] + vector[i]
            }
        }
        
        // Dividir pela quantidade de vetores
        val size = vectors.size
        for (i in result.indices) {
            result[i] = result[i] / size
        }
        
        return result
    }
    
    /**
     * Converte uma distância euclidiana para uma medida de similaridade entre 0 e 1.
     * Quanto maior a distância, menor a similaridade.
     *
     * @param distance Distância euclidiana
     * @return Valor de similaridade entre 0 e 1
     */
    fun euclideanToSimilarity(distance: Float): Float {
        return 1.0f / (1.0f + distance)
    }
}
