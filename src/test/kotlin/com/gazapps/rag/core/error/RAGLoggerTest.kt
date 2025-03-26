package com.gazapps.rag.core.error

import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.io.TempDir
import java.io.ByteArrayOutputStream
import java.io.PrintStream
import java.nio.file.Files
import java.nio.file.Path
import java.time.LocalDateTime

class RAGLoggerTest {

    @Test
    fun `ConsoleLogger should log messages at or above minimum level`() {
        val originalOut = System.out
        val originalErr = System.err
        
        try {
            // Redirect System.out and System.err
            val outContent = ByteArrayOutputStream()
            val errContent = ByteArrayOutputStream()
            System.setOut(PrintStream(outContent))
            System.setErr(PrintStream(errContent))
            
            // Create logger with INFO minimum level
            val logger = ConsoleLogger(LogLevel.INFO)
            
            // Log at different levels
            logger.debug("Debug message", "TestComponent")
            logger.info("Info message", "TestComponent")
            logger.warn("Warning message", "TestComponent")
            logger.error("Error message", "TestComponent")
            
            // Check output streams
            val outString = outContent.toString()
            val errString = errContent.toString()
            
            // Debug should not be logged
            assertFalse(outString.contains("Debug message"))
            
            // Info and warn should go to stdout
            assertTrue(outString.contains("INFO [TestComponent] Info message"))
            assertTrue(outString.contains("WARN [TestComponent] Warning message"))
            
            // Error should go to stderr
            assertTrue(errString.contains("ERROR [TestComponent] Error message"))
            
        } finally {
            // Restore original streams
            System.setOut(originalOut)
            System.setErr(originalErr)
        }
    }
    
    @Test
    fun `FileLogger should write logs to file`(@TempDir tempDir: Path) {
        val logFile = tempDir.resolve("test.log").toFile()
        
        // Create logger with DEBUG minimum level
        val logger = FileLogger(logFile.absolutePath, LogLevel.DEBUG)
        
        // Log at different levels
        logger.debug("Debug message", "TestComponent")
        logger.info("Info message", "TestComponent")
        logger.warn("Warning message", "TestComponent")
        logger.error("Error message", "TestComponent", Exception("Test exception"))
        
        // Read log file
        val logContent = logFile.readText()
        
        // Verify all log levels were written
        assertTrue(logContent.contains("DEBUG [TestComponent] Debug message"))
        assertTrue(logContent.contains("INFO [TestComponent] Info message"))
        assertTrue(logContent.contains("WARN [TestComponent] Warning message"))
        assertTrue(logContent.contains("ERROR [TestComponent] Error message"))
        assertTrue(logContent.contains("java.lang.Exception: Test exception"))
    }
    
    @Test
    fun `CompositeLogger should log to all loggers`() {
        // Create mock loggers
        val mockLogger1 = MockLogger()
        val mockLogger2 = MockLogger()
        
        // Create composite logger
        val compositeLogger = CompositeLogger(listOf(mockLogger1, mockLogger2))
        
        // Log at different levels
        compositeLogger.debug("Debug message", "TestComponent")
        compositeLogger.info("Info message", "TestComponent")
        compositeLogger.warn("Warning message", "TestComponent")
        compositeLogger.error("Error message", "TestComponent")
        
        // Verify both loggers received all messages
        assertEquals(4, mockLogger1.entries.size)
        assertEquals(4, mockLogger2.entries.size)
        
        // Verify message order and content
        assertEquals(LogLevel.DEBUG, mockLogger1.entries[0].level)
        assertEquals("Debug message", mockLogger1.entries[0].message)
        
        assertEquals(LogLevel.INFO, mockLogger2.entries[1].level)
        assertEquals("Info message", mockLogger2.entries[1].message)
    }
    
    @Test
    fun `Throwable extension should log exceptions`() {
        val mockLogger = MockLogger()
        RAGLoggerFactory.setLogger(mockLogger)
        
        // Create and log exception
        val exception = RuntimeException("Test exception")
        exception.log(LogLevel.ERROR, "TestComponent", "Custom message")
        
        // Verify log entry
        assertEquals(1, mockLogger.entries.size)
        val entry = mockLogger.entries.first()
        
        assertEquals(LogLevel.ERROR, entry.level)
        assertEquals("TestComponent", entry.component)
        assertEquals("Custom message", entry.message)
        assertSame(exception, entry.exception)
    }
    
    @Test
    fun `LogEntry format should properly format the log entry`() {
        val timestamp = LocalDateTime.of(2023, 1, 1, 12, 0, 0)
        val exception = RuntimeException("Test exception")
        val context = mapOf("key1" to "value1", "key2" to 42)
        
        val entry = LogEntry(
            timestamp = timestamp,
            level = LogLevel.ERROR,
            message = "Test message",
            component = "TestComponent",
            exception = exception,
            context = context
        )
        
        val formatted = entry.format()
        
        // Verify format
        assertTrue(formatted.contains("[2023-01-01 12:00:00.000]"))
        assertTrue(formatted.contains("ERROR"))
        assertTrue(formatted.contains("[TestComponent]"))
        assertTrue(formatted.contains("Test message"))
        assertTrue(formatted.contains("key1=value1"))
        assertTrue(formatted.contains("key2=42"))
        assertTrue(formatted.contains("java.lang.RuntimeException: Test exception"))
        assertTrue(formatted.contains("Stack trace:"))
    }
    
    @Test
    fun `RAGLoggerFactory should manage global logger instance`() {
        // Get default logger (should be ConsoleLogger)
        val defaultLogger = RAGLoggerFactory.getLogger()
        assertTrue(defaultLogger is ConsoleLogger)
        
        // Set custom logger
        val customLogger = MockLogger()
        RAGLoggerFactory.setLogger(customLogger)
        
        // Get logger again, should be our custom logger
        val currentLogger = RAGLoggerFactory.getLogger()
        assertSame(customLogger, currentLogger)
        
        // Create and use factory methods
        val consoleLogger = RAGLoggerFactory.createConsoleLogger(LogLevel.DEBUG)
        assertTrue(consoleLogger is ConsoleLogger)
        
        val fileLogger = RAGLoggerFactory.createFileLogger("test.log", LogLevel.INFO)
        assertTrue(fileLogger is FileLogger)
        
        val compositeLogger = RAGLoggerFactory.createCompositeLogger(listOf(consoleLogger, fileLogger))
        assertTrue(compositeLogger is CompositeLogger)
    }
    
    // Mock logger for testing
    private class MockLogger : RAGLogger {
        val entries = mutableListOf<LogEntry>()
        
        override fun log(entry: LogEntry) {
            entries.add(entry)
        }
    }
}
