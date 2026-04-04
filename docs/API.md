# Eridian API Documentation

Real-time 3D Gaussian Splatting Pipeline - Complete API Reference

---

## Table of Contents

- [Configuration](#configuration)
- [Logging](#logging)
- [Camera](#camera)
- [Depth Estimation](#depth-estimation)
- [SLAM](#slam)
- [Gaussian Splat](#gaussian-splat)
- [Spatial Data Structures](#spatial-data-structures)
- [Async Pipeline](#async-pipeline)
- [Examples](#examples)

---

## Configuration

### `Config` Class

Configuration management with YAML support and validation.

```python
from eridian import Config, get_config

# Load default configuration
config = get_config()

# Load from custom file
config = get_config("path/to/config.yaml")

# Access configuration values
print(config.camera.width)
print(config.depth.model_name)
print(config.splat.max_points)
```

#### Configuration Categories

**Camera:**
- `width` (int): Processing width in pixels
- `height` (int): Processing height in pixels
- `fov_deg` (float): Horizontal field of view in degrees
- `target_fps` (int): Target frames per second

**Depth:**
- `model` (str): Depth model name ("MiDaS_small", "DPT_Hybrid")
- `min_depth` (float): Minimum depth in meters
- `max_depth` (float): Maximum depth in meters
- `min_disparity` (float): Minimum normalized disparity (0-1)
- `max_disparity` (float): Maximum normalized disparity (0-1)

**Splat:**
- `max_points` (int): Maximum number of Gaussians to maintain
- `point_step` (int): Sample every Nth pixel (1-12)
- `output_dir` (str): Directory for output files

**Performance:**
- `save_interval` (int): Auto-save every N frames
- `backup_interval` (int): Create backup every N frames
- `downsample_ratio` (float): Target ratio when downsampling (0-1)
- `downsample_interval` (int): Check downsampling every N frames

**SLAM:**
- `max_features` (int): Maximum features to track
- `feature_quality` (float): Feature detection quality (0.01-1.0)
- `feature_min_distance` (int): Minimum distance between features
- `feature_refresh_interval` (int): Refresh features every N frames
- `ransac_threshold` (float): RANSAC threshold for pose estimation

---

## Logging

### `EridianLogger`

Structured logging with timestamps and component tags.

```python
from eridian import setup_logging, get_logger

# Setup logging
logger = setup_logging(level="INFO", console=True)

# Log messages
logger.debug("Debug information")
logger.info("General information")
logger.warning("Warning message")
logger.error("Error occurred", exc_info=True)
logger.critical("Critical error")
```

### `PerformanceMonitor`

Track performance metrics and timers.

```python
from eridian import get_monitor

monitor = get_monitor()

# Timer operations
monitor.start_timer("operation")
# ... do work ...
duration = monitor.stop_timer("operation")

# Counter operations
monitor.increment_counter("frames_processed")
count = monitor.get_counter("frames_processed")

# Log all counters
monitor.log_counters("Prefix: ")
```

### `ErrorHandler`

Centralized error handling with retry logic.

```python
from eridian import get_error_handler

error_handler = get_error_handler()

# Handle an error
error_handler.handle_error(
    error_type="camera_error",
    error=exception,
    context="frame_capture",
    critical=False
)

# Configure retry limits
error_handler.set_max_retries("camera_error", 5)
```

### Decorators

```python
from eridian import safe_execute, timer_context, error_context

# Safe execution decorator
@safe_execute("operation_name", default_return=None, critical=False)
def risky_function():
    # This function will be wrapped with error handling
    pass

# Timer context manager
with timer_context(logger, "Operation", log_threshold=0.1):
    # Code here will be timed
    pass

# Error context manager
with error_context(logger, error_handler, "error_type", context="operation"):
    # Errors here will be handled
    pass
```

---

## Camera

### `CameraManager`

Manage camera devices and frame capture.

```python
from eridian import CameraManager

# Create camera manager
camera = CameraManager(target_width=640, target_height=480)

# Scan for available devices
devices = camera.scan_devices()
for device in devices:
    print(f"{device.name}: {device.width}x{device.height}")

# Open best camera
success = camera.open_device(device_index=-1)  # -1 = auto-select

# Capture frames
while True:
    frame = camera.capture_frame()
    if frame is not None:
        rgb, gray, processed = camera.process_frame(frame)
        # Use rgb, gray, processed...

# Get camera intrinsics
K = camera.get_intrinsics(fov_deg=77.0)

# Close camera
camera.close()
```

### `CameraDevice`

Represents a camera device with metadata.

```python
# Properties:
device.index      # Camera index
device.width      # Image width
device.height     # Image height
device.name       # Device name
device.resolution # width * height
```

---

## Depth Estimation

### `MultiModelDepthEstimator`

Multi-model depth estimation with automatic selection.

```python
from eridian import MultiModelDepthEstimator, ModelRegistry

# Create estimator (auto-selects best model)
estimator = MultiModelDepthEstimator(
    preferred_model="midas_small",
    auto_select=True,
    max_depth=8.0,
    min_depth=0.1
)

# Estimate depth
depth_normalized = estimator.estimate(rgb_image)
depth_metric = estimator.estimate_metric(rgb_image)

# Switch models
estimator.switch_model("midas_hybrid")

# Benchmark all models
results = estimator.benchmark(rgb_image, iterations=10)

# Get model info
info = estimator.get_current_model_info()
```

### `ModelRegistry`

Register and manage depth estimation models.

```python
from eridian import ModelRegistry

# List all models
models = ModelRegistry.list_models()
for model in models:
    print(f"{model.display_name}: fast={model.fast}, accurate={model.accurate}")

# Query by category
fast_models = ModelRegistry.get_fast_models()
accurate_models = ModelRegistry.get_accurate_models()

# Get specific model
model = ModelRegistry.get_model("midas_small")
```

### Available Models

- **MiDaS Small**: Fast (~10ms), low memory (200MB)
- **MiDaS Hybrid**: Balanced (~50ms), moderate memory (800MB)
- **MiDaS Large**: Accurate (~100ms), high memory (1.5GB)

---

## SLAM

### `PoseTracker`

Camera pose tracking using visual odometry.

```python
from eridian import PoseTracker
import numpy as np

# Create pose tracker
K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64)
tracker = PoseTracker(K)

# Update pose from new frame
pose = tracker.update(gray_image, depth_map, max_depth=8.0)

# Get current state
position = tracker.get_position()  # (x, y, z)
orientation = tracker.get_orientation()  # 3x3 rotation matrix

# Reset to origin
tracker.reset()
```

### `SLAMPipeline`

Integrated SLAM pipeline with depth and pose tracking.

```python
from eridian import SLAMPipeline, MultiModelDepthEstimator

# Create components
depth_estimator = MultiModelDepthEstimator()
slam = SLAMPipeline(K, depth_estimator, config={
    'max_features': 500,
    'max_depth': 8.0
})

# Process frames
pose, depth = slam.process_frame(gray_image, rgb_image)

# Get trajectory
trajectory = slam.get_trajectory()  # Nx3 array of positions

# Reset
slam.reset()
```

---

## Gaussian Splat

### `SplatBuilder`

Build and manage 3D Gaussian splats.

```python
from eridian import SplatBuilder

# Create splat builder
builder = SplatBuilder(
    max_points=800000,
    point_step=4,
    output_dir="./splat"
)

# Add points from frame
builder.add(
    rgb=rgb_image,
    depth=depth_map,
    pose=pose_matrix,
    K=intrinsics_matrix,
    point_step=4,
    min_depth=0.05,
    max_depth=0.92,
    max_metric_depth=8.0
)

# Get arrays for visualization
positions, colors = builder.get_arrays()

# Save to PLY file
builder.save("output.ply")

# Get statistics
stats = builder.get_stats()
print(f"Points: {stats['total_points']}, Frames: {stats['frame_count']}")

# Estimate memory usage
memory_mb = builder.estimate_memory()

# Reset
builder.reset()
```

### `SplatManager`

Manage multiple splat files.

```python
from eridian import SplatManager

# Create manager
manager = SplatManager(output_dir="./splat")

# List all splats
splats = manager.list_splats()

# Get latest splat
latest = manager.get_latest_splat()

# Cleanup old files
manager.cleanup_old_splats(keep=5)
```

---

## Spatial Data Structures

### `SpatialHash`

Grid-based spatial partitioning for efficient queries.

```python
from eridian import SpatialHash

# Create spatial hash
sh = SpatialHash(cell_size=0.5)

# Insert single point
idx = sh.insert(position=np.array([1.0, 2.0, 3.0]))

# Insert multiple points
positions = np.random.randn(100, 3) * 5.0
indices = sh.insert_batch(positions)

# Radius query
neighbors = sh.query_radius(np.array([0.0, 0.0, 0.0]), radius=2.0)

# Find duplicates
duplicates = sh.find_duplicates(position, threshold=0.02)

# Get all positions
all_positions = sh.get_all_positions()

# Get statistics
stats = sh.get_stats()
```

### `Octree`

Hierarchical spatial partitioning for LOD rendering.

```python
from eridian import Octree

# Create octree
octree = Octree(
    center=np.zeros(3),
    size=10.0,
    max_points_per_node=1000,
    max_depth=8
)

# Insert points
for pos, color in zip(positions, colors):
    octree.insert(pos, color)

# Radius query
results = octree.query_radius(position, radius=2.0)

# Get all points
positions, colors = octree.get_all_points()

# Get statistics
stats = octree.get_stats()
print(f"Total points: {stats['total_points']}")
print(f"Max depth: {stats['max_depth']}")
```

---

## Async Pipeline

### `AsyncPipeline`

Concurrent frame processing with thread pools.

```python
from eridian import AsyncPipeline, PipelineStage

# Create pipeline
pipeline = AsyncPipeline(
    max_workers=4,
    queue_size=10,
    enable_visualization=True
)

# Register stage handlers
pipeline.register_stage(PipelineStage.DEPTH, depth_handler)
pipeline.register_stage(PipelineStage.POSE, pose_handler)
pipeline.register_stage(PipelineStage.SPLAT, splat_handler)

# Start pipeline
pipeline.start()

# Submit frames
rgb = ...  # RGB image
gray = ...  # Grayscale image
success = pipeline.submit_frame(rgb, gray, metadata={"frame_id": 1})

# Get processed output
output = pipeline.get_output(timeout=0.1)

# Get metrics
metrics = pipeline.get_metrics()
summary = metrics.get_summary()

# Get queue sizes
queue_sizes = pipeline.get_queue_sizes()

# Stop pipeline
pipeline.stop()
```

### `PipelineBuilder`

Fluent builder for pipeline construction.

```python
from eridian import PipelineBuilder

# Build pipeline
pipeline = PipelineBuilder() \
    .with_workers(4) \
    .with_queue_size(10) \
    .with_visualization(True) \
    .build()
```

### Pipeline Stages

- `CAPTURE`: Frame acquisition
- `DEPTH`: Depth estimation
- `POSE`: Pose tracking
- `SPLAT`: Gaussian splat construction
- `SAVE`: File persistence
- `VISUALIZE`: Real-time rendering

---

## Examples

### Basic Example

```python
from eridian import (
    setup_logging,
    CameraManager,
    MultiModelDepthEstimator,
    PoseTracker,
    SplatBuilder
)
import numpy as np

# Setup
logger = setup_logging(level="INFO")
camera = CameraManager(target_width=640, target_height=480)
depth_estimator = MultiModelDepthEstimator()
splat_builder = SplatBuilder(max_points=800000)

# Open camera
camera.open_device()
K = camera.get_intrinsics()
tracker = PoseTracker(K)

# Process frames
for i in range(100):
    frame = camera.capture_frame()
    if frame is None:
        break
    
    rgb, gray, _ = camera.process_frame(frame)
    
    # Estimate depth
    depth = depth_estimator.estimate(rgb)
    
    # Track pose
    pose = tracker.update(gray, depth)
    
    # Add to splat
    splat_builder.add(rgb, depth, pose, K)
    
    if i % 30 == 0:
        print(f"Frame {i}: {splat_builder.get_stats()}")

# Save result
splat_builder.save("output.ply")
camera.close()
```

### Advanced Example with Async Pipeline

```python
from eridian import (
    setup_logging,
    CameraManager,
    MultiModelDepthEstimator,
    SplatBuilder,
    AsyncPipeline,
    PipelineBuilder,
    PipelineStage,
    get_config
)

# Setup
config = get_config()
logger = setup_logging(level=config.logging.level)

# Create components
camera = CameraManager(
    target_width=config.camera.width,
    target_height=config.camera.height
)
depth_estimator = MultiModelDepthEstimator(
    preferred_model=config.depth.model,
    max_depth=config.depth.max_depth
)
splat_builder = SplatBuilder(
    max_points=config.splat.max_points,
    point_step=config.splat.point_step
)

# Create pipeline
pipeline = PipelineBuilder() \
    .with_workers(4) \
    .with_queue_size(10) \
    .build()

# Register handlers
def depth_handler(frame_data):
    frame_data.depth = depth_estimator.estimate(frame_data.rgb)
    return frame_data

def pose_handler(frame_data):
    K = camera.get_intrinsics(config.camera.fov_deg)
    frame_data.pose = tracker.update(frame_data.gray, frame_data.depth)
    return frame_data

def splat_handler(frame_data):
    K = camera.get_intrinsics(config.camera.fov_deg)
    splat_builder.add(
        frame_data.rgb,
        frame_data.depth,
        frame_data.pose,
        K
    )
    return frame_data

pipeline.register_stage(PipelineStage.DEPTH, depth_handler)
pipeline.register_stage(PipelineStage.POSE, pose_handler)
pipeline.register_stage(PipelineStage.SPLAT, splat_handler)

# Run
camera.open_device()
tracker = PoseTracker(camera.get_intrinsics())
pipeline.start()

try:
    for i in range(1000):
        frame = camera.capture_frame()
        if frame is None:
            break
        
        rgb, gray, _ = camera.process_frame(frame)
        pipeline.submit_frame(rgb, gray)
        
        if i % 30 == 0:
            splat_builder.save("splat.ply")
finally:
    pipeline.stop()
    camera.close()
    splat_builder.save("final.ply")
```

---

## Configuration File Format

```yaml
# config/default.yaml

camera:
  width: 640
  height: 480
  fov_deg: 77.0
  target_fps: 15

depth:
  model: "MiDaS_small"
  min_depth: 0.1
  max_depth: 8.0
  min_disparity: 0.05
  max_disparity: 0.92

splat:
  max_points: 800000
  point_step: 4
  output_dir: "./splat"

performance:
  save_interval: 30
  backup_interval: 150
  downsample_ratio: 0.8
  downsample_interval: 8

slam:
  max_features: 500
  feature_quality: 0.03
  feature_min_distance: 7
  feature_refresh_interval: 20
  ransac_threshold: 1.0

logging:
  level: "INFO"
  console: true
  fps_update_interval: 20
```

---

## Best Practices

1. **Always close resources**: Use `try/finally` or context managers
2. **Handle errors gracefully**: Use `safe_execute` decorator
3. **Monitor performance**: Use `PerformanceMonitor` for critical paths
4. **Configure appropriately**: Adjust settings based on hardware
5. **Use async pipeline** for maximum throughput
6. **Enable spatial hashing** for large scenes
7. **Choose appropriate depth model** based on accuracy/speed needs

---

## Performance Tips

- **For speed**: Use `MiDaS_small`, increase `point_step`, reduce `max_points`
- **For quality**: Use `DPT_Hybrid`, decrease `point_step`, increase `max_points`
- **For memory**: Enable spatial hashing, use smaller `save_interval`
- **For throughput**: Use async pipeline with appropriate `max_workers`

---

## Troubleshooting

### Low FPS
- Reduce `point_step` in configuration
- Use faster depth model (`MiDaS_small`)
- Reduce processing resolution
- Enable async pipeline

### Out of Memory
- Reduce `max_points`
- Enable spatial downsampling
- Reduce `queue_size` in async pipeline
- Use CPU instead of GPU if limited VRAM

### Poor Depth Quality
- Use better depth model (`DPT_Hybrid`)
- Improve lighting conditions
- Ensure textured surfaces
- Avoid mirrors and glass

### Pose Drift
- Move camera slowly
- Ensure good feature tracking
- Increase `max_features`
- Adjust `ransac_threshold`

---

## Version History

### v0.1.0
- Initial modular architecture
- Multi-model depth estimation
- Async pipeline support
- Spatial hashing and octree
- Comprehensive logging and error handling
- Configuration management

---

For more information, see the main README.md and run individual module tests with:
```bash
python -m eridian.module_name
```