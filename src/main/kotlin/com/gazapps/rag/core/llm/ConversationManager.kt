package com.gazapps.rag.core.llm

import com.gazapps.rag.core.ScoredDocument
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.decodeFromStream
import java.io.File
import java.time.Instant

/**
 * Message in a conversation
 */
@Serializable
data class ConversationMessage(
    val role: MessageRole,
    val content: String,
    val timestamp: Long = Instant.now().epochSecond,
    val metadata: Map<String, String> = emptyMap()
)

/**
 * Role of a message in a conversation
 */
enum class MessageRole {
    USER, ASSISTANT, SYSTEM
}

/**
 * A conversation history with messages and related documents
 */
@Serializable
data class Conversation(
    val id: String,
    val messages: MutableList<ConversationMessage> = mutableListOf(),
    val metadata: MutableMap<String, String> = mutableMapOf(),
    val documentIds: MutableSet<String> = mutableSetOf()
) {
    /**
     * Add a message to the conversation
     */
    fun addMessage(role: MessageRole, content: String, metadata: Map<String, String> = emptyMap()) {
        messages.add(ConversationMessage(role, content, Instant.now().epochSecond, metadata))
    }
    
    /**
     * Add a user message to the conversation
     */
    fun addUserMessage(content: String, metadata: Map<String, String> = emptyMap()) {
        addMessage(MessageRole.USER, content, metadata)
    }
    
    /**
     * Add an assistant message to the conversation
     */
    fun addAssistantMessage(content: String, metadata: Map<String, String> = emptyMap()) {
        addMessage(MessageRole.ASSISTANT, content, metadata)
    }
    
    /**
     * Add a system message to the conversation
     */
    fun addSystemMessage(content: String, metadata: Map<String, String> = emptyMap()) {
        addMessage(MessageRole.SYSTEM, content, metadata)
    }
    
    /**
     * Track documents associated with this conversation
     */
    fun addDocuments(documents: List<ScoredDocument>) {
        documentIds.addAll(documents.map { it.document.id })
    }
    
    /**
     * Get the last N messages
     */
    fun getLastMessages(n: Int): List<ConversationMessage> {
        return messages.takeLast(n.coerceAtMost(messages.size))
    }
    
    /**
     * Format the conversation history as a prompt for an LLM
     */
    fun formatHistoryAsPrompt(
        maxMessages: Int = 10,
        includeSystem: Boolean = true,
        historyPromptTemplate: String = "Previous conversation:\n{history}\n\nNew question: {question}"
    ): String {
        val messagesToInclude = if (includeSystem) {
            messages.takeLast(maxMessages)
        } else {
            messages.filter { it.role != MessageRole.SYSTEM }.takeLast(maxMessages)
        }
        
        if (messagesToInclude.isEmpty()) return ""
        
        val lastQuestion = if (messagesToInclude.last().role == MessageRole.USER) {
            messagesToInclude.last().content
        } else {
            ""
        }
        
        val historyText = messagesToInclude.dropLast(if (lastQuestion.isNotEmpty()) 1 else 0)
            .joinToString("\n\n") { message -> 
                "${message.role.name.lowercase()}: ${message.content}"
            }
        
        return historyPromptTemplate
            .replace("{history}", historyText)
            .replace("{question}", lastQuestion)
    }
}

/**
 * Manages conversation histories
 */
class ConversationManager(
    private val storageDirectory: File? = null
) {
    private val conversations = mutableMapOf<String, Conversation>()
    
    init {
        storageDirectory?.let {
            if (!it.exists()) {
                it.mkdirs()
            }
        }
    }
    
    /**
     * Create a new conversation
     */
    fun createConversation(id: String = generateConversationId()): Conversation {
        val conversation = Conversation(id)
        conversations[id] = conversation
        return conversation
    }
    
    /**
     * Get a conversation by ID
     */
    fun getConversation(id: String): Conversation? {
        return conversations[id] ?: loadConversation(id)
    }
    
    /**
     * Get or create a conversation
     */
    fun getOrCreateConversation(id: String): Conversation {
        return getConversation(id) ?: createConversation(id)
    }
    
    /**
     * Save a conversation to disk
     */
    fun saveConversation(conversation: Conversation) {
        storageDirectory?.let {
            val file = File(it, "${conversation.id}.json")
            file.writeText(Json.encodeToString(conversation))
        }
        
        conversations[conversation.id] = conversation
    }
    
    /**
     * Load a conversation from disk
     */
    private fun loadConversation(id: String): Conversation? {
        storageDirectory?.let {
            val file = File(it, "$id.json")
            if (file.exists()) {
                return try {
                    val conversation = Json.decodeFromStream<Conversation>(file.inputStream())
                    conversations[id] = conversation
                    conversation
                } catch (e: Exception) {
                    null
                }
            }
        }
        return null
    }
    
    /**
     * Generate a unique conversation ID
     */
    private fun generateConversationId(): String {
        return "conv-${Instant.now().epochSecond}-${(Math.random() * 10000).toInt()}"
    }
}
