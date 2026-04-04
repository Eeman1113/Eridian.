"""
Eridian - Real-time 3D Gaussian Splatting Pipeline
Map the world in 3D. One frame at a time.
"""

__version__ = "0.1.0"
__author__ = "Eeman Majumder"

from .config import Config, get_config, reset_config
from .logging import (
    EridianLogger,
    PerformanceMonitor,
    ErrorHandler,
    setup_logging,
    get_logger,
    get_monitor,
    get_error_handler,
    safe_execute,
    timer_context,
    error_context,
)

__all__ = [
    'Config',
    'get_config', 
    'reset_config',
    'EridianLogger',
    'PerformanceMonitor',
    'ErrorHandler',
    'setup_logging',
    'get_logger',
    'get_monitor',
    'get_error_handler',
    'safe_execute',
    'timer_context',
    'error_context',
]