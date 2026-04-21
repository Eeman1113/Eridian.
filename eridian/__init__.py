"""Eridian - Real-time monocular 3D point cloud reconstruction.

GCT mode powered by LingBot-MAP (https://github.com/Robbyant/lingbot-map)
Geometric Context Transformer for Streaming 3D Reconstruction
License: Apache 2.0 | Authors: Robbyant Team
"""

from eridian.main import (
    # High-level API
    Eridian,
    EridianConfig,
    FrameResult,
    # Low-level components
    CameraCapture,
    DepthEstimator,
    PoseEstimator,
    PointCloud,
    Visualizer3D,
    WorldMapper,
    # Utilities
    save_ply,
    load_ply,
    probe_cameras,
    pick_camera,
    render_demo_video,
    cli_main,
    # Constants
    DEPTH_MODELS,
)

# GCT backend (lazy import — only loaded when --gct is used)
def _lazy_gct():
    from eridian.gct_backend import GCTReconstructor, gct_to_pointcloud
    return GCTReconstructor, gct_to_pointcloud

__version__ = "0.3.0"

__all__ = [
    # High-level API
    "Eridian",
    "EridianConfig",
    "FrameResult",
    # Low-level components
    "CameraCapture",
    "DepthEstimator",
    "PoseEstimator",
    "PointCloud",
    "Visualizer3D",
    "WorldMapper",
    # Utilities
    "save_ply",
    "load_ply",
    "probe_cameras",
    "pick_camera",
    "render_demo_video",
    "cli_main",
    # Constants
    "DEPTH_MODELS",
    # GCT backend
    "GCTReconstructor",
    "gct_to_pointcloud",
]
