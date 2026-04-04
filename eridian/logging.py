#!/usr/bin/env python3
"""
Logging and Error Handling Module for Eridian
Provides structured logging, error tracking, and performance monitoring.
"""

import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Optional, Callable, Any
from functools import wraps
from contextlib import contextmanager


class EridianLogger:
    """Custom logger for Eridian with enhanced formatting and error handling."""
    
    def __init__(self, name: str = "eridian", level: str = "INFO", console: bool = True):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # Clear any existing handlers
        self.logger.handlers.clear()
        
        # Console handler
        if console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(getattr(logging, level.upper()))
            
            # Custom formatter with timestamps and component tags
            formatter = logging.Formatter(
                '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
                datefmt='%H:%M:%S'
            )
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # File handler (optional - can be enabled)
        self._file_handler: Optional[logging.FileHandler] = None
    
    def enable_file_logging(self, log_file: str):
        """Enable logging to a file."""
        try:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            self._file_handler = logging.FileHandler(log_file)
            self._file_handler.setLevel(self.logger.level)
            
            formatter = logging.Formatter(
                '[%(asctime)s] [%(name)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            self._file_handler.setFormatter(formatter)
            self.logger.addHandler(self._file_handler)
            
            self.info(f"File logging enabled: {log_file}")
        except Exception as e:
            self.error(f"Failed to enable file logging: {e}")
    
    def debug(self, msg: str, **kwargs):
        """Log debug message."""
        self.logger.debug(msg, **kwargs)
    
    def info(self, msg: str, **kwargs):
        """Log info message."""
        self.logger.info(msg, **kwargs)
    
    def warning(self, msg: str, **kwargs):
        """Log warning message."""
        self.logger.warning(msg, **kwargs)
    
    def error(self, msg: str, exc_info: bool = False, **kwargs):
        """Log error message."""
        self.logger.error(msg, exc_info=exc_info, **kwargs)
    
    def critical(self, msg: str, exc_info: bool = True, **kwargs):
        """Log critical error message."""
        self.logger.critical(msg, exc_info=exc_info, **kwargs)
    
    def exception(self, msg: str, **kwargs):
        """Log exception with traceback."""
        self.logger.exception(msg, **kwargs)


class PerformanceMonitor:
    """Monitor and log performance metrics."""
    
    def __init__(self, logger: EridianLogger):
        self.logger = logger
        self._timers: dict[str, float] = {}
        self._counters: dict[str, int] = {}
    
    def start_timer(self, name: str):
        """Start a named timer."""
        self._timers[name] = time.time()
    
    def stop_timer(self, name: str, log_threshold: Optional[float] = None) -> float:
        """Stop a named timer and optionally log if it exceeds threshold."""
        if name not in self._timers:
            self.logger.warning(f"Timer '{name}' was not started")
            return 0.0
        
        elapsed = time.time() - self._timers[name]
        del self._timers[name]
        
        if log_threshold is None or elapsed >= log_threshold:
            self.logger.debug(f"Timer '{name}': {elapsed*1000:.1f}ms")
        
        return elapsed
    
    def increment_counter(self, name: str, delta: int = 1):
        """Increment a named counter."""
        self._counters[name] = self._counters.get(name, 0) + delta
    
    def get_counter(self, name: str) -> int:
        """Get the value of a named counter."""
        return self._counters.get(name, 0)
    
    def reset_counter(self, name: str):
        """Reset a named counter."""
        if name in self._counters:
            del self._counters[name]
    
    def log_counters(self, prefix: str = ""):
        """Log all counters with optional prefix."""
        if not self._counters:
            return
        
        self.logger.debug(f"{prefix}Counters: {self._counters}")


class ErrorHandler:
    """Centralized error handling and recovery."""
    
    def __init__(self, logger: EridianLogger):
        self.logger = logger
        self._error_counts: dict[str, int] = {}
        self._max_retries: dict[str, int] = {}
    
    def handle_error(self, error_type: str, error: Exception, 
                    context: str = "", critical: bool = False) -> bool:
        """
        Handle an error with logging and optional recovery.
        
        Args:
            error_type: Type/category of error
            error: The exception that occurred
            context: Additional context about where the error occurred
            critical: Whether this is a critical error that should stop execution
            
        Returns:
            True if error was handled and execution should continue, False otherwise
        """
        error_key = f"{error_type}:{context}"
        self._error_counts[error_key] = self._error_counts.get(error_key, 0) + 1
        
        count = self._error_counts[error_key]
        max_retries = self._max_retries.get(error_type, 3)
        
        if critical:
            self.logger.critical(
                f"Critical error in {context}: {error}",
                exc_info=True
            )
            return False
        
        if count > max_retries:
            self.logger.error(
                f"Too many errors ({count}) for {error_type} in {context}: {error}"
            )
            return False
        
        self.logger.warning(
            f"Error {count}/{max_retries} in {context}: {error}"
        )
        
        return True
    
    def set_max_retries(self, error_type: str, max_retries: int):
        """Set maximum retries for a specific error type."""
        self._max_retries[error_type] = max_retries
    
    def get_error_count(self, error_type: str, context: str = "") -> int:
        """Get the error count for a specific error type and context."""
        error_key = f"{error_type}:{context}"
        return self._error_counts.get(error_key, 0)


def safe_execute(error_type: str, default_return: Any = None, 
                critical: bool = False, context: str = ""):
    """
    Decorator for safe execution with error handling.
    
    Args:
        error_type: Type/category of error for logging
        default_return: Value to return if an error occurs
        critical: Whether errors should be treated as critical
        context: Additional context for error messages
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = kwargs.get('logger') or (
                args[0].logger if hasattr(args[0], 'logger') else None
            )
            error_handler = kwargs.get('error_handler') or (
                args[0].error_handler if hasattr(args[0], 'error_handler') else None
            )
            
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_context = f"{context} in {func.__name__}"
                
                if error_handler:
                    should_continue = error_handler.handle_error(
                        error_type, e, error_context, critical
                    )
                    if should_continue:
                        return default_return
                
                if logger:
                    if critical:
                        logger.critical(f"Error in {error_context}: {e}", exc_info=True)
                    else:
                        logger.error(f"Error in {error_context}: {e}", exc_info=True)
                
                return default_return
        
        return wrapper
    return decorator


@contextmanager
def timer_context(logger: EridianLogger, name: str, 
                 log_threshold: Optional[float] = None):
    """Context manager for timing code blocks."""
    start = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start
        if log_threshold is None or elapsed >= log_threshold:
            logger.debug(f"{name}: {elapsed*1000:.1f}ms")


@contextmanager
def error_context(logger: EridianLogger, error_handler: ErrorHandler,
                 error_type: str, context: str = "", critical: bool = False):
    """Context manager for error handling."""
    try:
        yield
    except Exception as e:
        should_continue = error_handler.handle_error(
            error_type, e, context, critical
        )
        if not should_continue:
            raise


# Global logger instance
_logger: Optional[EridianLogger] = None
_monitor: Optional[PerformanceMonitor] = None
_error_handler: Optional[ErrorHandler] = None


def setup_logging(level: str = "INFO", console: bool = True, 
                 log_file: Optional[str] = None) -> EridianLogger:
    """Setup global logging system."""
    global _logger, _monitor, _error_handler
    
    _logger = EridianLogger(level=level, console=console)
    _monitor = PerformanceMonitor(_logger)
    _error_handler = ErrorHandler(_logger)
    
    if log_file:
        _logger.enable_file_logging(log_file)
    
    return _logger


def get_logger() -> EridianLogger:
    """Get the global logger instance."""
    if _logger is None:
        return setup_logging()
    return _logger


def get_monitor() -> PerformanceMonitor:
    """Get the global performance monitor."""
    if _monitor is None:
        setup_logging()
    return _monitor


def get_error_handler() -> ErrorHandler:
    """Get the global error handler."""
    if _error_handler is None:
        setup_logging()
    return _error_handler


if __name__ == "__main__":
    # Test the logging module
    logger = setup_logging(level="DEBUG")
    monitor = get_monitor()
    error_handler = get_error_handler()
    
    logger.info("Logging system test started")
    
    # Test performance monitoring
    with timer_context(logger, "Test operation", log_threshold=0.0):
        time.sleep(0.1)
    
    monitor.increment_counter("test_counter")
    monitor.increment_counter("test_counter")
    monitor.log_counters("Test: ")
    
    # Test error handling
    @safe_execute("test_error", default_return=None, context="test function")
    def test_function():
        raise ValueError("This is a test error")
    
    result = test_function()
    logger.info(f"Function returned: {result}")
    
    logger.info("Logging system test completed ✓")