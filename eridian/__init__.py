"""Eridian - Real-time monocular 3D point cloud reconstruction."""

from eridian.main import (
    # High-level API
    Eridian,
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
)

__version__ = "0.1.7"

__all__ = [
    # High-level API
    "Eridian",
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
]
