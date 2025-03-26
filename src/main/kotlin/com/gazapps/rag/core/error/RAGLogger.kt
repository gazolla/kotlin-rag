package com.gazapps.rag.core.error

import java.time.LocalDateTime
import java.time.format.DateTimeFormatter

/**
 * Log level enum
 */
enum class LogLevel {
    DEBUG, INFO, WARN, ERROR, NONE
}

/**
 * Log entry data class
 */
data class LogEntry(
    val timestamp: LocalDateTime = LocalDateTime.now(),
    val level: LogLevel,
    val message: String,
    val component: String,
    val exception: Throwable? = null,
    val context: Map<String, Any?> = emptyMap()
) {
    fun format(): String {
        val formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss.SSS")
        val timestampStr = timestamp.format(formatter)
        val contextStr = if (context.isNotEmpty()) {
            " | " + context.entries.joinToString(", ") { "${it.key}=${it.value}" }
        } else {
            ""
        }
        
        val exceptionStr = exception?.let {
            "\n  ${it.javaClass.name}: ${it.message ?: "No message"}" +
            "\n  Stack trace: ${it.stackTraceToString().lines().take(3).joinToString("\n    ")}"
        } ?: ""
        
        return "[$timestampStr] ${level.name} [$component] $message$contextStr$exceptionStr"
    }
}

/**
 * Logger interface for the RAG library
 */
interface RAGLogger {
    fun log(entry: LogEntry)
    
    fun debug(message: String, component: String, exception: Throwable? = null, context: Map<String, Any?> = emptyMap()) {
        log(LogEntry(level = LogLevel.DEBUG, message = message, component = component, exception = exception, context = context))
    }
    
    fun info(message: String, component: String, exception: Throwable? = null, context: Map<String, Any?> = emptyMap()) {
        log(LogEntry(level = LogLevel.INFO, message = message, component = component, exception = exception, context = context))
    }
    
    fun warn(message: String, component: String, exception: Throwable? = null, context: Map<String, Any?> = emptyMap()) {
        log(LogEntry(level = LogLevel.WARN, message = message, component = component, exception = exception, context = context))
    }
    
    fun error(message: String, component: String, exception: Throwable? = null, context: Map<String, Any?> = emptyMap()) {
        log(LogEntry(level = LogLevel.ERROR, message = message, component = component, exception = exception, context = context))
    }
}

/**
 * Console logger implementation
 */
class ConsoleLogger(private val minLevel: LogLevel = LogLevel.INFO) : RAGLogger {
    override fun log(entry: LogEntry) {
        if (entry.level.ordinal >= minLevel.ordinal) {
            val formattedEntry = entry.format()
            when (entry.level) {
                LogLevel.ERROR -> System.err.println(formattedEntry)
                else -> println(formattedEntry)
            }
        }
    }
}

/**
 * File logger implementation
 */
class FileLogger(
    private val filePath: String,
    private val minLevel: LogLevel = LogLevel.INFO,
    private val append: Boolean = true
) : RAGLogger {
    private val file = java.io.File(filePath)
    
    init {
        // Create parent directories if they don't exist
        file.parentFile?.mkdirs()
        
        // Clear file if not appending
        if (!append && file.exists()) {
            file.writeText("")
        }
    }
    
    override fun log(entry: LogEntry) {
        if (entry.level.ordinal >= minLevel.ordinal) {
            file.appendText(entry.format() + "\n")
        }
    }
}

/**
 * Composite logger that logs to multiple destinations
 */
class CompositeLogger(private val loggers: List<RAGLogger>) : RAGLogger {
    override fun log(entry: LogEntry) {
        loggers.forEach { it.log(entry) }
    }
}

/**
 * Singleton logger manager for the RAG library
 */
object RAGLoggerFactory {
    private var logger: RAGLogger = ConsoleLogger(LogLevel.INFO)
    
    fun getLogger(): RAGLogger = logger
    
    fun setLogger(newLogger: RAGLogger) {
        logger = newLogger
    }
    
    fun createConsoleLogger(minLevel: LogLevel = LogLevel.INFO): RAGLogger {
        return ConsoleLogger(minLevel)
    }
    
    fun createFileLogger(filePath: String, minLevel: LogLevel = LogLevel.INFO, append: Boolean = true): RAGLogger {
        return FileLogger(filePath, minLevel, append)
    }
    
    fun createCompositeLogger(loggers: List<RAGLogger>): RAGLogger {
        return CompositeLogger(loggers)
    }
}

// Extension function to easily log exceptions
fun Throwable.log(level: LogLevel = LogLevel.ERROR, component: String, message: String = this.message ?: "Error occurred", context: Map<String, Any?> = emptyMap()) {
    val logger = RAGLoggerFactory.getLogger()
    when (level) {
        LogLevel.DEBUG -> logger.debug(message, component, this, context)
        LogLevel.INFO -> logger.info(message, component, this, context)
        LogLevel.WARN -> logger.warn(message, component, this, context)
        LogLevel.ERROR -> logger.error(message, component, this, context)
        LogLevel.NONE -> {} // Do nothing
    }
}
