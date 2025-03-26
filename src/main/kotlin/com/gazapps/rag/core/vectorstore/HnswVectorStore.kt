package com.gazapps.rag.core.vectorstore

import com.gazapps.rag.core.Document
import com.gazapps.rag.core.ScoredDocument
import com.gazapps.rag.core.VectorStore
import com.gazapps.rag.core.vectorstore.SimilarityUtils.SimilarityMetric
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.slf4j.LoggerFactory
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.CopyOnWriteArrayList
import java.util.concurrent.locks.ReentrantReadWriteLock
import kotlin.concurrent.read
import kotlin.concurrent.write
import kotlin.random.Random

/**
 * In-memory vector store with HNSW (Hierarchical Navigable Small World) indexing
 * for fast approximate nearest neighbor search.
 *
 * This implementation provides significantly faster search performance than
 * a brute-force approach for large numbers of vectors.
 *
 * @param m Number of connections per node (default: 16)
 * @param efConstruction Quality of graph construction (default: 200)
 * @param efSearch Quality of search (higher = more accurate, default: 100)
 * @param similarityMetric The metric to use for similarity calculations
 */
class HnswVectorStore(
    private val m: Int = 16,
    private val efConstruction: Int = 200,
    private val efSearch: Int = 100,
    private val similarityMetric: SimilarityMetric = SimilarityMetric.COSINE
) : VectorStore {
    
    private val logger = LoggerFactory.getLogger(HnswVectorStore::class.java)
    
    // Document storage
    private val documents = ConcurrentHashMap<String, Document>()
    private val embeddings = ConcurrentHashMap<String, FloatArray>()
    
    // Lock for graph structure modifications
    private val lock = ReentrantReadWriteLock()
    
    // HNSW graph structure
    private val maxLevel = 6  // Max level in the graph
    private val layerMult = 1 / Math.log(m.toDouble()) // Level probability multiplier
    private val layers = Array(maxLevel + 1) { ConcurrentHashMap<String, MutableList<String>>() }
    private var entryPoint: String? = null
    
    // Node metadata storage
    private val nodeLevels = ConcurrentHashMap<String, Int>()
    
    override suspend fun store(document: Document, embedding: FloatArray) = withContext(Dispatchers.Default) {
        val id = document.id
        
        documents[id] = document
        
        // Normalize embedding if using cosine similarity
        val normalizedEmbedding = if (similarityMetric == SimilarityMetric.COSINE) {
            SimilarityUtils.normalize(embedding)
        } else {
            embedding
        }
        
        embeddings[id] = normalizedEmbedding
        
        // Insert into HNSW graph
        insertIntoGraph(id, normalizedEmbedding)
        
        logger.debug("Stored document with ID: $id")
    }
    
    override suspend fun batchStore(documents: List<Document>, embeddings: List<FloatArray>) = withContext(Dispatchers.Default) {
        require(documents.size == embeddings.size) {
            "Number of documents (${documents.size}) must match number of embeddings (${embeddings.size})"
        }
        
        for (i in documents.indices) {
            val document = documents[i]
            val embedding = embeddings[i]
            store(document, embedding)
        }
        
        logger.debug("Batch stored ${documents.size} documents")
    }
    
    override suspend fun search(
        query: FloatArray, 
        limit: Int, 
        filter: Map<String, Any>?
    ): List<ScoredDocument> = withContext(Dispatchers.Default) {
        // Normalize query if using cosine similarity
        val normalizedQuery = if (similarityMetric == SimilarityMetric.COSINE) {
            SimilarityUtils.normalize(query)
        } else {
            query
        }
        
        // Use HNSW search
        val searchResults = searchHnsw(normalizedQuery, limit * 2, efSearch)
        
        // Apply filters and prepare ScoredDocument objects
        val filteredResults = searchResults
            .asSequence()
            .mapNotNull { (id, score) -> 
                val document = documents[id] ?: return@mapNotNull null
                
                // Apply filter if provided
                if (filter != null && !MetadataFilter.matchesSimpleFilter(document, filter)) {
                    return@mapNotNull null
                }
                
                ScoredDocument(document, score)
            }
            .sortedByDescending { it.score }
            .take(limit)
            .toList()
        
        return@withContext filteredResults
    }
    
    override suspend fun delete(documentId: String) = withContext(Dispatchers.Default) {
        lock.write {
            // Delete connections to this node
            for (layer in 0..maxLevel) {
                layers[layer].forEach { (nodeId, connections) ->
                    connections.remove(documentId)
                }
                
                // Remove node's connections
                layers[layer].remove(documentId)
            }
            
            // Update entry point if needed
            if (entryPoint == documentId) {
                entryPoint = if (documents.isEmpty()) null
                else documents.keys.firstOrNull()
            }
            
            // Remove from internal data structures
            nodeLevels.remove(documentId)
            documents.remove(documentId)
            embeddings.remove(documentId)
            
            logger.debug("Deleted document with ID: $documentId")
        }
    }
    
    override suspend fun clear() = withContext(Dispatchers.Default) {
        lock.write {
            documents.clear()
            embeddings.clear()
            nodeLevels.clear()
            
            for (layer in 0..maxLevel) {
                layers[layer].clear()
            }
            
            entryPoint = null
            
            logger.info("Cleared all documents")
        }
    }
    
    /**
     * Get the number of documents in the store
     */
    fun size(): Int = documents.size
    
    /**
     * Insert a document into the HNSW graph
     */
    private fun insertIntoGraph(id: String, embedding: FloatArray) {
        lock.write {
            // Generate random level for new node
            val level = generateNodeLevel()
            nodeLevels[id] = level
            
            // If this is the first node, make it the entry point
            if (entryPoint == null) {
                entryPoint = id
                
                // Initialize empty connections
                for (l in 0..level) {
                    layers[l][id] = CopyOnWriteArrayList()
                }
                
                return@write
            }
            
            // Start from entry point
            var currObj = entryPoint!!
            var currObjLevel = nodeLevels[currObj] ?: 0
            
            // For each level, find the closest node to connect to
            for (lc in maxLevel downTo level + 1) {
                if (lc <= currObjLevel) {
                    val closestNode = findClosestOnLayer(embedding, currObj, lc)
                    if (calculateSimilarity(embedding, embeddings[closestNode]!!) > 
                       calculateSimilarity(embedding, embeddings[currObj]!!)) {
                        currObj = closestNode
                        currObjLevel = nodeLevels[currObj] ?: 0
                    }
                }
            }
            
            // For each level from the node's level down to 0
            for (lc in minOf(level, currObjLevel) downTo 0) {
                // Find closest nodes at this level
                val neighbors = findNeighbors(embedding, currObj, lc, efConstruction)
                
                // Initialize connections for new node at this level
                if (!layers[lc].containsKey(id)) {
                    layers[lc][id] = CopyOnWriteArrayList()
                }
                
                // Connect to neighbors (bidirectional)
                connectNodes(id, neighbors, lc)
                
                // Get new closest node for next level
                if (neighbors.isNotEmpty()) {
                    currObj = neighbors.first().first
                }
            }
            
            // Update entry point if new node is at a higher level
            if (level > nodeLevels[entryPoint ?: ""] ?: 0) {
                entryPoint = id
            }
        }
    }
    
    /**
     * Connect a node to its neighbors
     */
    private fun connectNodes(nodeId: String, neighbors: List<Pair<String, Float>>, level: Int) {
        val connections = layers[level].getOrPut(nodeId) { CopyOnWriteArrayList() }
        
        // Add best neighbors up to M connections
        val candidateNeighbors = neighbors.take(m).map { it.first }
        connections.addAll(candidateNeighbors)
        
        // Add bidirectional connections
        for (neighborId in candidateNeighbors) {
            val neighborConnections = layers[level].getOrPut(neighborId) { CopyOnWriteArrayList() }
            neighborConnections.add(nodeId)
            
            // Prune connections if necessary
            if (neighborConnections.size > m) {
                pruneConnections(neighborId, level)
            }
        }
    }
    
    /**
     * Prune connections to keep only the best M
     */
    private fun pruneConnections(nodeId: String, level: Int) {
        val connections = layers[level][nodeId] ?: return
        if (connections.size <= m) return
        
        val nodeEmbedding = embeddings[nodeId] ?: return
        
        // Calculate similarity to all connections
        val scoredConnections = connections.mapNotNull { connId ->
            val connEmbedding = embeddings[connId] ?: return@mapNotNull null
            Pair(connId, calculateSimilarity(nodeEmbedding, connEmbedding))
        }
        
        // Keep only the best M connections
        val bestConnections = scoredConnections
            .sortedByDescending { it.second }
            .take(m)
            .map { it.first }
        
        // Update connections
        layers[level][nodeId] = CopyOnWriteArrayList(bestConnections)
    }
    
    /**
     * Find the closest neighbors to a query embedding
     */
    private fun findNeighbors(
        query: FloatArray, 
        entryNodeId: String, 
        level: Int,
        ef: Int
    ): List<Pair<String, Float>> {
        // Visited nodes set
        val visited = hashSetOf<String>()
        
        // Initialize with entry node
        val entryEmbedding = embeddings[entryNodeId] ?: return emptyList()
        val entryScore = calculateSimilarity(query, entryEmbedding)
        
        // Candidates (nodes to be expanded), ordered by similarity
        val candidates = sortedSetOf<Pair<Float, String>>(
            compareByDescending<Pair<Float, String>> { it.first }
                .thenBy { it.second }
        )
        candidates.add(Pair(entryScore, entryNodeId))
        
        // Results, ordered by similarity
        val results = sortedSetOf<Pair<Float, String>>(
            compareByDescending<Pair<Float, String>> { it.first }
                .thenBy { it.second }
        )
        results.add(Pair(entryScore, entryNodeId))
        
        visited.add(entryNodeId)
        
        // Main search loop
        while (candidates.isNotEmpty()) {
            // Get the closest candidate
            val current = candidates.first()
            candidates.remove(current)
            
            // If we've found enough results and the next candidate is worse than our worst result
            if (results.size >= ef && current.first < results.last().first) {
                break
            }
            
            // Get connections of the current node at this level
            val connections = layers[level][current.second] ?: continue
            
            // Examine each connection
            for (connId in connections) {
                if (connId in visited) continue
                
                visited.add(connId)
                
                // Calculate similarity
                val connEmbedding = embeddings[connId] ?: continue
                val similarity = calculateSimilarity(query, connEmbedding)
                
                // If our results set isn't full yet, or this connection is better than our worst result
                if (results.size < ef || similarity > results.last().first) {
                    candidates.add(Pair(similarity, connId))
                    results.add(Pair(similarity, connId))
                    
                    // Remove worst results if we've found too many
                    if (results.size > ef) {
                        results.remove(results.last())
                    }
                }
            }
        }
        
        // Convert to list of ID and similarity score
        return results.map { Pair(it.second, it.first) }
    }
    
    /**
     * Find the closest node to a query embedding on a specific layer
     */
    private fun findClosestOnLayer(query: FloatArray, startNodeId: String, level: Int): String {
        var current = startNodeId
        var currentSimilarity = calculateSimilarity(query, embeddings[current] ?: return current)
        var changed = true
        
        // Greedy search on this layer
        while (changed) {
            changed = false
            
            // Get connections at this level
            val connections = layers[level][current] ?: break
            
            // Check all connections for a better node
            for (connId in connections) {
                val connEmbedding = embeddings[connId] ?: continue
                val similarity = calculateSimilarity(query, connEmbedding)
                
                if (similarity > currentSimilarity) {
                    current = connId
                    currentSimilarity = similarity
                    changed = true
                    break
                }
            }
        }
        
        return current
    }
    
    /**
     * Perform HNSW search for approximate nearest neighbors
     */
    private fun searchHnsw(query: FloatArray, k: Int, ef: Int): List<Pair<String, Float>> {
        lock.read {
            val entryPoint = this.entryPoint ?: return emptyList()
            
            // Start from the entry point
            var currObj = entryPoint
            var currObjLevel = nodeLevels[currObj] ?: 0
            
            // For each level, find the closest node
            for (lc in maxLevel downTo 1) {
                if (lc <= currObjLevel) {
                    val closestNode = findClosestOnLayer(query, currObj, lc)
                    val closestEmbedding = embeddings[closestNode] ?: continue
                    val currEmbedding = embeddings[currObj] ?: continue
                    
                    if (calculateSimilarity(query, closestEmbedding) > 
                       calculateSimilarity(query, currEmbedding)) {
                        currObj = closestNode
                        currObjLevel = nodeLevels[currObj] ?: 0
                    }
                }
            }
            
            // Search on level 0 with full ef parameter
            val neighbors = findNeighbors(query, currObj, 0, ef)
            
            // Return the top k results
            return neighbors.take(k)
        }
    }
    
    /**
     * Generate a random level for a new node
     */
    private fun generateNodeLevel(): Int {
        val r = -Math.log(Random.nextDouble()) * layerMult
        return minOf(maxLevel, r.toInt())
    }
    
    /**
     * Calculate similarity between two vectors
     */
    private fun calculateSimilarity(a: FloatArray, b: FloatArray): Float {
        return SimilarityUtils.calculateSimilarity(a, b, similarityMetric)
    }
}
