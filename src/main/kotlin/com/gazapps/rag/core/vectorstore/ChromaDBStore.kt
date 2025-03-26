package com.gazapps.rag.core.vectorstore

import kotlin.ranges.ClosedRange

import com.gazapps.rag.core.Document
import com.gazapps.rag.core.SimpleDocument
import com.gazapps.rag.core.ScoredDocument
import com.gazapps.rag.core.VectorStore
import io.ktor.client.*
import io.ktor.client.call.*
import io.ktor.client.engine.cio.*
import io.ktor.client.plugins.contentnegotiation.*
import io.ktor.client.request.*
import io.ktor.client.statement.*
import io.ktor.http.*
import io.ktor.serialization.kotlinx.json.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.*
import org.slf4j.LoggerFactory
import kotlin.math.min

/**
 * Implementation of VectorStore that uses ChromaDB for storing and retrieving embeddings.
 * ChromaDB is an open-source embedding database that can be self-hosted or used as a managed service.
 *
 * @param baseUrl The URL of the ChromaDB instance
 * @param collectionName The name of the collection to use
 * @param embeddingDimension The dimension of the embedding vectors (needed for collection creation)
 * @param distance The distance metric to use (default: "cosine")
 * @param createCollection Whether to create the collection if it doesn't exist (default: true)
 */
class ChromaDBStore(
    private val baseUrl: String,
    private val collectionName: String,
    private val embeddingDimension: Int = 1536,
    private val distance: String = "cosine",
    private val createCollection: Boolean = true,
    private val httpClient: HttpClient = HttpClient(CIO) {
        install(ContentNegotiation) {
            json(Json {
                ignoreUnknownKeys = true
                prettyPrint = false
                isLenient = true
            })
        }
    }
) : VectorStore {
    
    private val logger = LoggerFactory.getLogger(ChromaDBStore::class.java)
    
    // Track documents for retrieval (ChromaDB doesn't store the full document content)
    private val documentCache = mutableMapOf<String, Document>()
    
    init {
        if (createCollection) {
            runBlocking {
                ensureCollectionExists()
            }
        }
    }
    
    /**
     * Ensure that the collection exists in ChromaDB
     */
    private suspend fun ensureCollectionExists() {
        try {
            // Check if collection exists
            val collectionsResponse = httpClient.get("$baseUrl/api/v1/collections")
            val collectionsJson: Map<String, JsonElement> = collectionsResponse.body()
            
            val collections = collectionsJson["collections"]?.jsonArray ?: JsonArray(emptyList())
            
            val collectionExists = collections.any { 
                it.jsonObject["name"]?.jsonPrimitive?.content == collectionName 
            }
            
            if (!collectionExists) {
                // Create the collection
                val createRequest = mapOf(
                    "name" to collectionName,
                    "metadata" to mapOf<String, String>(),
                    "get_or_create" to true
                )
                
                httpClient.post("$baseUrl/api/v1/collections") {
                    contentType(ContentType.Application.Json)
                    setBody(createRequest)
                }
                
                logger.info("Created collection: $collectionName")
            } else {
                logger.info("Collection already exists: $collectionName")
            }
        } catch (e: Exception) {
            logger.error("Error ensuring collection exists: ${e.message}", e)
            throw RuntimeException("Failed to initialize ChromaDB collection", e)
        }
    }
    
    override suspend fun store(document: Document, embedding: FloatArray) {
        try {
            val embeddingList = embedding.toList()
            
            val request = mapOf(
                "ids" to listOf(document.id),
                "embeddings" to listOf(embeddingList),
                "metadatas" to listOf(document.metadata),
                "documents" to listOf(document.content)
            )
            
            val response = httpClient.post("$baseUrl/api/v1/collections/$collectionName/points") {
                contentType(ContentType.Application.Json)
                setBody(request)
            }
            
            if (response.status.isSuccess()) {
                // Cache document for later retrieval
                documentCache[document.id] = document
                logger.debug("Stored document with ID: ${document.id}")
            } else {
                logger.error("Failed to store document. Status: ${response.status}")
                throw RuntimeException("Failed to store document: ${response.bodyAsText()}")
            }
        } catch (e: Exception) {
            logger.error("Error storing document: ${e.message}", e)
            throw RuntimeException("Failed to store document in ChromaDB", e)
        }
    }
    
    override suspend fun batchStore(documents: List<Document>, embeddings: List<FloatArray>) {
        require(documents.size == embeddings.size) {
            "Number of documents must match number of embeddings"
        }
        
        if (documents.isEmpty()) return
        
        // ChromaDB has a limit on batch size, so we process in chunks
        val batchSize = 100
        
        for (i in documents.indices step batchSize) {
            val batchEnd = min(i + batchSize, documents.size)
            val batch = documents.subList(i, batchEnd)
            val batchEmbeddings = embeddings.subList(i, batchEnd)
            
            try {
                val ids = batch.map { it.id }
                val embeddingsList = batchEmbeddings.map { it.toList() }
                val metadatas = batch.map { it.metadata }
                val contents = batch.map { it.content }
                
                val request = mapOf(
                    "ids" to ids,
                    "embeddings" to embeddingsList,
                    "metadatas" to metadatas,
                    "documents" to contents
                )
                
                val response = httpClient.post("$baseUrl/api/v1/collections/$collectionName/points") {
                    contentType(ContentType.Application.Json)
                    setBody(request)
                }
                
                if (response.status.isSuccess()) {
                    // Cache documents for later retrieval
                    batch.forEach { documentCache[it.id] = it }
                    logger.debug("Stored batch of ${batch.size} documents")
                } else {
                    logger.error("Failed to store batch. Status: ${response.status}")
                    throw RuntimeException("Failed to store batch: ${response.bodyAsText()}")
                }
            } catch (e: Exception) {
                logger.error("Error storing batch: ${e.message}", e)
                throw RuntimeException("Failed to store batch in ChromaDB", e)
            }
        }
    }
    
    override suspend fun search(query: FloatArray, limit: Int, filter: Map<String, Any>?): List<ScoredDocument> {
        try {
            val embeddingList = query.toList()
            
            // Convert filter to ChromaDB where format if provided
            val whereClause = filter?.let { 
                buildWhereClause(it)
            }
            
            val request = buildMap<String, Any> {
                put("query_embeddings", listOf(embeddingList))
                put("n_results", limit)
                put("include", listOf("documents", "metadatas", "distances"))
                
                if (whereClause != null) {
                    put("where", whereClause)
                }
            }
            
            val response = httpClient.post("$baseUrl/api/v1/collections/$collectionName/query") {
                contentType(ContentType.Application.Json)
                setBody(request)
            }
            
            if (response.status.isSuccess()) {
                // Parse response
                val responseBody: JsonObject = response.body()
                
                val ids = responseBody["ids"]?.jsonArray?.firstOrNull()?.jsonArray?.map { it.jsonPrimitive.content } ?: emptyList()
                val contents = responseBody["documents"]?.jsonArray?.firstOrNull()?.jsonArray?.map { it.jsonPrimitive.content } ?: emptyList()
                val metadatas = responseBody["metadatas"]?.jsonArray?.firstOrNull()?.jsonArray?.map { 
                    it.jsonObject.mapValues { (_, value) -> 
                        when (value) {
                            is JsonPrimitive -> {
                                when {
                                    value.isString -> value.content
                                    value.intOrNull != null -> value.int
                                    value.longOrNull != null -> value.long
                                    value.doubleOrNull != null -> value.double
                                    value.booleanOrNull != null -> value.boolean
                                    else -> value.content
                                }
                            }
                            else -> value.toString()
                        }
                    }
                } ?: emptyList()
                
                val distances = responseBody["distances"]?.jsonArray?.firstOrNull()?.jsonArray?.map { it.jsonPrimitive.float } ?: emptyList()
                
                // Prepare results with a more explicit approach
                val results = mutableListOf<ScoredDocument>()
                
                // Process results if we have matching lengths
                if (ids.size == contents.size && ids.size == metadatas.size && ids.size == distances.size) {
                    for (i in ids.indices) {
                        val docId = ids[i]
                        val docContent = contents[i]
                        val docMetadata = metadatas[i]
                        val distanceValue = distances[i]
                        
                        // Use cached document if available, otherwise create a new one
                        val document = documentCache[docId] ?: SimpleDocument(docId, docContent, docMetadata)
                        
                        // Convert distance to similarity score
                        val similarityScore = 1f - distanceValue  // Distance to similarity conversion
                        
                        results.add(ScoredDocument(document, similarityScore))
                    }
                } else {
                    logger.warn("Mismatched result arrays: ids=${ids.size}, contents=${contents.size}, metadatas=${metadatas.size}, distances=${distances.size}")
                }
                
                return results
            } else {
                logger.error("Search failed. Status: ${response.status}")
                throw RuntimeException("Search failed: ${response.bodyAsText()}")
            }
        } catch (e: Exception) {
            logger.error("Error during search: ${e.message}", e)
            throw RuntimeException("Failed to search in ChromaDB", e)
        }
    }
    
    override suspend fun delete(documentId: String) {
        try {
            val request = mapOf("ids" to listOf(documentId))
            
            val response = httpClient.delete("$baseUrl/api/v1/collections/$collectionName/points") {
                contentType(ContentType.Application.Json)
                setBody(request)
            }
            
            if (response.status.isSuccess()) {
                documentCache.remove(documentId)
                logger.debug("Deleted document with ID: $documentId")
            } else {
                logger.error("Failed to delete document. Status: ${response.status}")
                throw RuntimeException("Failed to delete document: ${response.bodyAsText()}")
            }
        } catch (e: Exception) {
            logger.error("Error deleting document: ${e.message}", e)
            throw RuntimeException("Failed to delete document from ChromaDB", e)
        }
    }
    
    override suspend fun clear() {
        try {
            // ChromaDB doesn't have a direct "clear" method, so we delete the collection and recreate it
            httpClient.delete("$baseUrl/api/v1/collections/$collectionName")
            
            if (createCollection) {
                ensureCollectionExists()
            }
            
            documentCache.clear()
            logger.info("Cleared collection: $collectionName")
        } catch (e: Exception) {
            logger.error("Error clearing collection: ${e.message}", e)
            throw RuntimeException("Failed to clear ChromaDB collection", e)
        }
    }
    
    /**
     * Close the HTTP client
     */
    fun close() {
        httpClient.close()
    }
    
    /**
     * Get the count of documents in the collection
     */
    suspend fun count(): Int {
        try {
            val response = httpClient.get("$baseUrl/api/v1/collections/$collectionName") {
                contentType(ContentType.Application.Json)
            }
            
            val responseBody: JsonObject = response.body()
            return responseBody["count"]?.jsonPrimitive?.int ?: 0
        } catch (e: Exception) {
            logger.error("Error getting collection count: ${e.message}", e)
            throw RuntimeException("Failed to get count from ChromaDB", e)
        }
    }
    
    /**
     * Build where clause for ChromaDB from a simple filter map
     */
    private fun buildWhereClause(filter: Map<String, Any>): Map<String, Any> {
        // Convert to ChromaDB where format
        return filter.mapValues { (_, value) ->
            when (value) {
                is String -> value
                is Number -> value
                is Boolean -> value
                is Collection<*> -> mapOf("\$in" to value)
                is ClosedRange<*> -> {
                    val start = (value.start as? Number)?.toDouble()
                    val end = (value.endInclusive as? Number)?.toDouble()
                    
                    if (start != null && end != null) {
                        mapOf("\$gte" to start, "\$lte" to end)
                    } else {
                        value.toString()
                    }
                }
                else -> value.toString()
            }
        }
    }
    
    private fun runBlocking(block: suspend () -> Unit) {
        kotlinx.coroutines.runBlocking {
            block()
        }
    }
}