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
from .camera import CameraManager, CameraDevice
from .depth import DepthEstimator, DepthEstimatorFactory
from .depth_enhanced import (
    MultiModelDepthEstimator,
    ModelRegistry,
    ModelSpec,
    BaseDepthEstimator,
)
from .slam import PoseTracker, SLAMPipeline
from .splat import SplatBuilder, SplatManager
from .spatial import SpatialHash, Octree, OctreeNode
from .pipeline import (
    AsyncPipeline,
    PipelineBuilder,
    PipelineStage,
    PipelineMetrics,
    FrameData,
)

__all__ = [
    # Configuration
    'Config',
    'get_config', 
    'reset_config',
    
    # Logging
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
    
    # Camera
    'CameraManager',
    'CameraDevice',
    
    # Depth
    'DepthEstimator',
    'DepthEstimatorFactory',
    'MultiModelDepthEstimator',
    'ModelRegistry',
    'ModelSpec',
    'BaseDepthEstimator',
    
    # SLAM
    'PoseTracker',
    'SLAMPipeline',
    
    # Splat
    'SplatBuilder',
    'SplatManager',
    
    # Spatial
    'SpatialHash',
    'Octree',
    'OctreeNode',
    
    # Pipeline
    'AsyncPipeline',
    'PipelineBuilder',
    'PipelineStage',
    'PipelineMetrics',
    'FrameData',
]