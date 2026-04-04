# Eridian
### Map the world in 3D. One frame at a time.

A real-time 3D reconstruction system that builds Gaussian splats from live monocular camera input. Inspired by tactile world modeling concepts, designed to eventually enable haptic feedback for visually impaired users.

---

## What This Does

Eridian captures real-time video from your camera, estimates depth per pixel, tracks camera pose, and incrementally builds a 3D Gaussian Splat representation of your environment.

```
Camera Input (USB/Continuity Camera)
        |
        v
  Depth Estimation (Multi-Model)  -->  Per-pixel depth map
        |
        v
  Pose Tracking (SLAM)           -->  Camera pose (R|t) in world space
        |
        v
  Gaussian Splat Builder          -->  Colored 3D gaussians in world coords
        |
        +-->  Real-time Viewer (Rerun)
        |
        +-->  Atomic Save (PLY format)
```

Each pixel from the depth map becomes a 3D Gaussian with:
  - Position  -- back-projected from camera using depth + intrinsics
  - Color     -- sampled from the RGB frame
  - Scale     -- sized proportionally to depth and pixel footprint
  - Opacity   -- higher confidence for closer objects
  - Rotation  -- identity (future: align to surface normals)

---

## New in Version 0.1.0

### Modular Architecture
- Completely refactored into separate, reusable modules
- Clear separation of concerns (camera, depth, SLAM, splat, spatial)
- Easy to test, extend, and integrate into other projects
- Type-safe interfaces throughout

### Configuration System
- YAML-based configuration with sensible defaults
- Type-safe configuration classes using dataclasses
- Runtime configuration validation
- Easy parameter tuning without code changes

### Multi-Model Depth Estimation
- Support for multiple depth networks (MiDaS Small, Hybrid, Large)
- Automatic model selection based on hardware
- Runtime model switching without restart
- Built-in benchmarking for performance comparison
- Model registry system for adding custom networks

### Enhanced Logging and Error Handling
- Structured logging with component-based filtering
- Performance monitoring with timers and counters
- Centralized error handling with retry logic
- Decorator-based error handling for clean code
- Comprehensive error tracking and reporting

### Spatial Indexing
- Spatial hash for O(1) nearest neighbor lookups
- Octree for hierarchical spatial partitioning
- Efficient duplicate detection and removal
- Memory-efficient storage for large point clouds
- Support for level-of-detail rendering

### Async Pipeline
- Concurrent frame processing with thread pools
- Separate worker threads for each pipeline stage
- Configurable concurrency and queue sizes
- Non-blocking frame submission
- Automatic flow control and backpressure management

### Comprehensive Documentation
- Complete API reference (docs/API.md)
- Architecture documentation (docs/ARCHITECTURE.md)
- Code examples and usage patterns
- Best practices and troubleshooting guides

### Testing Infrastructure
- Unit tests for core components
- Test harnesses in each module
- Validation of functionality without hardware
- Continuous integration ready

---

## Quick Start

```bash
# Clone and setup
git clone https://github.com/Eeman1113/Eridian..git
cd Eridian.
python3 setup.py

# Run
python3 run.py
```

First run creates a virtual environment, installs dependencies, downloads depth models, and starts. Every run after that launches instantly. The Rerun viewer opens automatically.

### Using the Modular API

```python
from eridian import (
    setup_logging,
    CameraManager,
    MultiModelDepthEstimator,
    PoseTracker,
    SplatBuilder,
    get_config
)
import numpy as np

# Setup
logger = setup_logging(level="INFO")
config = get_config()

# Create components
camera = CameraManager(
    target_width=config.camera.width,
    target_height=config.camera.height
)
depth_estimator = MultiModelDepthEstimator(
    preferred_model=config.depth.model
)
splat_builder = SplatBuilder(
    max_points=config.splat.max_points,
    output_dir=config.splat.output_dir
)

# Open camera
camera.open_device()
K = camera.get_intrinsics(config.camera.fov_deg)
tracker = PoseTracker(K)

# Process frames
for i in range(100):
    frame = camera.capture_frame()
    if frame is None:
        break
    
    rgb, gray, _ = camera.process_frame(frame)
    depth = depth_estimator.estimate(rgb)
    pose = tracker.update(gray, depth)
    splat_builder.add(rgb, depth, pose, K)
    
    if i % 30 == 0:
        print(f"Frame {i}: {splat_builder.get_stats()}")

# Save result
splat_builder.save("output.ply")
camera.close()
```

---

## Why Rerun instead of Open3D?

`open3d` has no Python 3.12/3.13 wheels and is effectively unmaintained on Apple Silicon. Rerun is a modern replacement:

  - Supports Python 3.8 through 3.13 on macOS, Linux, Windows
  - Shows 3D point cloud + RGB feed + depth map in one tabbed viewer
  - Logs the camera trajectory automatically
  - Zero configuration — viewer spawns in a browser tab or native app
  - Actively maintained and modern architecture

---

## Controls (camera window focused)

```
  Q / Esc  --  Quit and save final splat
  S        --  Force save right now
  R        --  Reset map + pose (start fresh)
  D        --  Toggle depth visualization overlay in the capture window
  +        --  Denser point cloud (slower)
  -        --  Sparser point cloud (faster)
```

---

## Configuration

Edit `config/default.yaml` to customize behavior:

```yaml
# Camera Settings
camera:
  width: 640
  height: 480
  fov_deg: 77.0
  target_fps: 15

# Depth Estimation
depth:
  model: "MiDaS_small"  # or "DPT_Hybrid", "DPT_Large"
  min_depth: 0.1
  max_depth: 8.0

# Point Cloud / Gaussian Splat
splat:
  max_points: 800000
  point_step: 4  # Lower = denser, higher = faster
  output_dir: "./splat"

# Performance & Memory
performance:
  save_interval: 30
  backup_interval: 150
  downsample_ratio: 0.8

# SLAM / Pose Tracking
slam:
  max_features: 500
  feature_quality: 0.03
  ransac_threshold: 1.0
```

Or use a custom config file:
```bash
python3 run.py --config path/to/custom.yaml
```

---

## Viewing Your Splat

The .ply files saved to ./splat/ are in standard 3DGS format and open in:

```
  SuperSplat  -->  https://supersplat.playcanvas.com  (drag and drop)
  Luma AI     -->  https://lumalabs.ai/interactive-scenes
  nerfstudio  -->  ns-viewer --load-config ...
  Three.js    -->  THREE.GaussianSplatMesh
```

---

## Performance Tips

```
  Too slow (< 5 FPS)    -->  Press - to reduce density
                              Set point_step: 8 in config
                              Use "MiDaS_small" model
                              Enable async pipeline

  Better depth quality  -->  Use "DPT_Hybrid" model
                              Increase camera resolution
                              Decrease point_step

  GPU / M1 / M2 Mac     -->  Auto-detected, check "[Depth] Device:" on startup
                              CUDA and MPS backends supported

  Too many points       -->  Lower max_points in config
                              Enable spatial hashing
                              Increase downsample_ratio

  Rerun viewer lag      -->  Normal for large point clouds
                              Live PLY is always up to date
                              Reduce viewer_update_interval
```

---

## Available Depth Models

### MiDaS Small
- Speed: ~10ms on GPU, ~30ms on CPU
- Memory: ~200MB
- Use case: Real-time applications, limited resources

### MiDaS Hybrid (DPT_Hybrid)
- Speed: ~50ms on GPU, ~80ms on CPU
- Memory: ~800MB
- Use case: Balanced quality and speed

### MiDaS Large (DPT_Large)
- Speed: ~100ms on GPU, ~150ms on CPU
- Memory: ~1.5GB
- Use case: Maximum quality, offline processing

Model selection is automatic based on available GPU memory, or can be specified in configuration.

---

## How to Get Best Results

```
  (1)  Move slowly -- fast motion blurs depth estimates
  (2)  Overlap coverage -- revisit areas from different angles
  (3)  Good lighting -- depth networks need texture and contrast
  (4)  Avoid mirrors and glass -- they confuse depth networks
  (5)  Start from a fixed point -- origin is your first frame
  (6)  Use appropriate model -- choose speed vs quality
  (7)  Configure for hardware -- adjust resolution and point_step
```

---

## Architecture

### Module Overview

**eridian.config**: Configuration management with YAML support
- Type-safe configuration classes
- Validation and defaults
- Runtime configuration loading

**eridian.logging**: Logging and error handling
- Structured logging with component filtering
- Performance monitoring (timers, counters)
- Centralized error handling with retries
- Decorators for clean error handling

**eridian.camera**: Camera device management
- Automatic device detection and selection
- Resolution scaling for performance
- Thread-safe capture interface
- Graceful disconnect handling

**eridian.depth**: Multi-model depth estimation
- MiDaS models (Small, Hybrid, Large)
- Automatic model selection based on hardware
- Multi-backend support (CUDA, MPS, CPU)
- Model registry for extensibility

**eridian.slam**: Camera pose tracking
- Lucas-Kanade optical flow
- Essential matrix pose estimation
- Scale estimation from depth
- Integrated SLAM pipeline

**eridian.splat**: Gaussian splat construction
- Incremental point accumulation
- Automatic downsampling
- Atomic PLY file saves
- Memory usage estimation

**eridian.spatial**: Spatial data structures
- Spatial hash for O(1) queries
- Octree for hierarchical partitioning
- Duplicate detection
- Memory-efficient storage

**eridian.pipeline**: Async processing pipeline
- Concurrent frame processing
- Thread pool management
- Bounded queues for flow control
- Performance metrics

### Data Flow

```
Camera Capture
    ↓
Async Pipeline (Stage 1: Capture)
    ↓
Depth Estimation (Stage 2: Depth)
    ↓
Pose Tracking (Stage 3: Pose)
    ↓
Splat Building (Stage 4: Splat)
    ↓
Save/Visualize (Stage 5: Output)
```

### Threading Model

- Main thread: Camera capture, UI coordination
- Worker threads: Depth estimation, pose tracking, splat building, save/visualize
- Thread-safe queues for inter-stage communication
- Locks for shared data structures

---

## Documentation

- **README.md**: This file - quick start and overview
- **docs/API.md**: Complete API reference with examples
- **docs/ARCHITECTURE.md**: System architecture and design details
- **config/default.yaml**: Default configuration with comments
- Inline documentation: Every module has docstrings and examples

---

## Testing

Run unit tests:
```bash
python -m pytest tests/
```

Run specific module tests:
```bash
python -m eridian.config
python -m eridian.camera
python -m eridian.depth
python -m eridian.splat
python -m eridian.spatial
python -m eridian.pipeline
```

---

## The Rocky Vision

This is a foundation for tactile world modeling. The path from here to a real tactile device:

```
  [Eridian]                          [Future]

  Real-time 3D map  ----------->  Extract surface mesh
         |                                |
         |                    Surface normals + gradients
         |                                |
         +-------------------------->  Haptic actuator array
                                          |
                                 Vibration = depth
                                 Texture   = surface roughness
                                 Shape     = gaussian curvature
```

Next steps toward the device:

```
  (1)  Loop closure    -- ORB-SLAM3 or COLMAP to fix drift
  (2)  Mesh extract    -- Poisson reconstruction from the splat
  (3)  Haptic encode   -- depth + normals -> motor / actuator signals
  (4)  Streaming API   -- real-time point cloud over WebSocket to a glove
  (5)  Object layer    -- "chair 0.8m ahead, slightly left"
```

---

## Honest Limitations

  - Not true 3DGS optimization. Real Gaussian Splatting needs SfM camera
    poses and thousands of gradient descent steps. This is an approximation.

  - Monocular depth scale is relative. Absolute metric scale is estimated,
    not measured. iPhone LiDAR would be dramatically better.

  - Pose drift. Without loop closure the map drifts.
    Works best for short captures (~30–60 seconds).

  - Depth network performs best in textured, well-lit environments.

  For a production blind-assist device the target stack is:
  iPhone LiDAR + ARKit + true SLAM.
  This is the fastest path to a working prototype.

---

## Project Structure

```
  eridian/           -- Core Python package
    __init__.py     -- Package initialization
    config.py        -- Configuration management
    logging.py       -- Logging and error handling
    camera.py        -- Camera device management
    depth.py         -- Depth estimation (basic)
    depth_enhanced.py -- Multi-model depth estimation
    slam.py          -- Pose tracking and SLAM
    splat.py         -- Gaussian splat builder
    spatial.py       -- Spatial hashing and octree
    pipeline.py      -- Async pipeline manager

  config/            -- Configuration files
    default.yaml     -- Default configuration

  docs/              -- Documentation
    API.md          -- Complete API reference
    ARCHITECTURE.md -- System architecture

  tests/             -- Unit tests
    test_core.py    -- Core component tests

  run.py             -- Original single-file implementation
  setup.py           -- Setup script
  requirements.txt    -- Python dependencies
  README.md          -- This file
  splat/             -- Output directory (created on first run)
    splat.ply            -- Live file, updated periodically
    splat_final.ply      -- Written on clean exit
    splat_000150.ply     -- Timestamped backups
```

---

## Dependencies

### Core Dependencies
- Python 3.8+
- PyTorch 2.0+
- OpenCV 4.8+
- NumPy 1.24+
- Rerun SDK 0.10+
- PyYAML 6.0+

### Optional Dependencies
- CUDA Toolkit (NVIDIA GPUs)
- Metal (Apple Silicon)

### System Requirements
- macOS (Apple Silicon preferred) or Linux
- 8GB+ RAM recommended
- GPU recommended but not required

---

## Roadmap

### Phase 1: Foundation (Complete)
- Modular architecture
- Configuration system
- Multi-model depth estimation
- Async pipeline
- Spatial indexing
- Comprehensive documentation

### Phase 2: Enhancements (In Progress)
- GPU acceleration for SLAM
- Better scale estimation
- Integration with main run.py
- Performance optimization

### Phase 3: Advanced Features (Planned)
- Loop closure integration
- Real-time mesh extraction
- Multi-camera support
- LiDAR integration
- Cloud processing backup

### Phase 4: Haptic Interface (Future)
- Surface mesh extraction
- Haptic encoding algorithms
- Streaming API for devices
- Object recognition layer

---

## Contributing

Contributions are welcome! Please see the documentation for:
- API reference: docs/API.md
- Architecture: docs/ARCHITECTURE.md
- Module examples: See __main__ sections in each module

When contributing:
- Follow existing code style and patterns
- Add tests for new functionality
- Update documentation
- Ensure all tests pass

---

## License

See LICENSE file for details.

---

## Support

- GitHub: https://github.com/Eeman1113/Eridian
- Issues: Report bugs and feature requests
- Documentation: See docs/ directory

---

Built with care for real-time 3D reconstruction and accessibility.
