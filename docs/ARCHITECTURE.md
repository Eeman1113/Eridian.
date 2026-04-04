# Eridian Architecture

Real-time 3D Gaussian Splatting Pipeline - System Architecture

---

## Overview

Eridian is a modular, real-time 3D reconstruction system that builds Gaussian splats from live monocular camera input. The architecture is designed for performance, extensibility, and ease of use.

```
┌─────────────────────────────────────────────────────────────┐
│                         Application Layer                    │
│                      (run.py, CLI, UI)                       │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                      Pipeline Manager                        │
│                   (eridian.pipeline)                         │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐     │
│  │ Capture │→ │  Depth  │→ │   Pose  │→ │  Splat  │     │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘     │
└─────────────────────────┬───────────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
┌───────▼───────┐ ┌──────▼──────┐ ┌──────▼──────┐
│   Camera      │ │    Depth    │ │    SLAM     │
│   Manager     │ │ Estimator   │ │   Pipeline  │
└───────────────┘ └─────────────┘ └─────────────┘
        │                 │                 │
┌───────▼─────────────────▼─────────────────▼───────┐
│              Spatial Indexing Layer                │
│           (SpatialHash, Octree)                    │
└───────────────┬───────────────────────────────────┘
                │
┌───────────────▼───────────────────────────────────┐
│            Infrastructure Layer                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐     │
│  │ Config   │ │  Logging  │ │  Error    │     │
│  │ Manager  │ │  System   │ │ Handler  │     │
│  └──────────┘  └──────────┘  └──────────┘     │
└──────────────────────────────────────────────────┘
```

---

## Core Components

### 1. Configuration System

**Purpose**: Centralized configuration management with YAML support.

**Key Classes**:
- `Config`: Main configuration container
- `CameraConfig`, `DepthConfig`, etc.: Type-safe sub-configurations

**Design Decisions**:
- YAML-based for human readability
- Dataclasses for type safety
- Validation on load
- Default values for all settings

**Dependencies**: None (infrastructure)

---

### 2. Logging System

**Purpose**: Structured logging, performance monitoring, error handling.

**Key Classes**:
- `EridianLogger`: Structured logger with timestamps
- `PerformanceMonitor`: Timing and counter tracking
- `ErrorHandler`: Centralized error handling with retries

**Design Decisions**:
- Component-based logging for better filtering
- Decorator-based error handling
- Non-blocking performance monitoring
- Thread-safe operations

**Dependencies**: None (infrastructure)

---

### 3. Camera Manager

**Purpose**: Abstract camera device management and frame capture.

**Key Classes**:
- `CameraManager`: Device detection and capture
- `CameraDevice`: Device metadata

**Design Decisions**:
- Auto-detection of best camera
- Resolution scaling for performance
- Thread-safe capture interface
- Graceful disconnect handling

**Dependencies**: Logging

**External Dependencies**: OpenCV

---

### 4. Depth Estimation

**Purpose**: Monocular depth estimation using neural networks.

**Key Classes**:
- `MultiModelDepthEstimator`: Multi-model depth estimation
- `ModelRegistry`: Model registration and management
- `BaseDepthEstimator`: Abstract base for estimators

**Design Decisions**:
- Model registry for extensibility
- Auto-selection based on hardware
- Multi-backend support (CUDA, MPS, CPU)
- Benchmarking capabilities

**Dependencies**: Logging, Config

**External Dependencies**: PyTorch, torch.hub

**Supported Models**:
- MiDaS Small (fast)
- MiDaS Hybrid (balanced)
- MiDaS Large (accurate)

---

### 5. SLAM Pipeline

**Purpose**: Camera pose tracking using visual odometry.

**Key Classes**:
- `PoseTracker`: Optical flow + essential matrix
- `SLAMPipeline`: Integrated depth + pose tracking

**Design Decisions**:
- Lucas-Kanade optical flow
- RANSAC for robustness
- Scale estimation from depth
- Feature refresh for long sequences

**Dependencies**: Logging, Depth

**External Dependencies**: OpenCV, NumPy

**Limitations**:
- Monocular scale ambiguity
- Drift over time
- No loop closure

---

### 6. Gaussian Splat Builder

**Purpose**: Incremental construction of 3D Gaussian splats.

**Key Classes**:
- `SplatBuilder`: Point accumulation and PLY export
- `SplatManager`: File management

**Design Decisions**:
- Thread-safe point accumulation
- Automatic downsampling
- Atomic file writes
- Memory usage estimation

**Dependencies**: Logging, Config

**External Dependencies**: NumPy

**Output Format**: 3DGS PLY format

---

### 7. Spatial Indexing

**Purpose**: Efficient spatial queries and memory management.

**Key Classes**:
- `SpatialHash`: Grid-based spatial partitioning
- `Octree`: Hierarchical spatial partitioning
- `OctreeNode`: Tree node implementation

**Design Decisions**:
- O(1) average lookup (SpatialHash)
- Hierarchical LOD (Octree)
- Thread-safe operations
- Duplicate detection

**Dependencies**: Logging

**External Dependencies**: NumPy

---

### 8. Async Pipeline

**Purpose**: Concurrent frame processing for maximum throughput.

**Key Classes**:
- `AsyncPipeline`: Pipeline manager with thread pools
- `PipelineBuilder`: Fluent pipeline construction
- `PipelineMetrics`: Performance monitoring

**Design Decisions**:
- Separate worker threads per stage
- Bounded queues for flow control
- Configurable concurrency
- Graceful shutdown

**Dependencies**: All modules

**Stages**:
1. Capture (input)
2. Depth estimation
3. Pose tracking
4. Splat construction
5. Save/visualize (output)

---

## Data Flow

### Frame Processing Flow

```
1. Camera Capture
   └─> FrameData {rgb, gray, metadata}

2. Depth Estimation
   └─> FrameData {rgb, gray, depth, metadata}

3. Pose Tracking
   └─> FrameData {rgb, gray, depth, pose, metadata}

4. Splat Construction
   └─> Update SplatBuilder (points accumulated)

5. Save/Visualize
   └─> Atomic PLY save, real-time viewer update
```

### Memory Flow

```
Camera Frame (RAM)
    ↓
Resized Frame (RAM)
    ↓
Depth Map (GPU/CPU)
    ↓
Pose Matrix (RAM)
    ↓
3D Points (RAM)
    ↓
Splat Storage (RAM + optional disk)
```

---

## Threading Model

### Main Thread
- Camera capture
- UI/CLI interaction
- Pipeline coordination

### Worker Threads (Async Pipeline)
- Depth estimation (1 thread)
- Pose tracking (1 thread)
- Splat building (1 thread)
- Save/visualize (1 thread)

### Thread Safety
- Thread-safe queues between stages
- Locks for shared data structures
- Atomic file operations
- Non-blocking metrics collection

---

## Performance Characteristics

### Bottlenecks (in order of impact)
1. Depth estimation (most expensive)
2. Pose tracking
3. Splat building (with many points)
4. Camera capture
5. File I/O (periodic)

### Optimization Strategies
- **Async Pipeline**: Parallelize independent stages
- **Model Selection**: Choose appropriate depth model
- **Spatial Indexing**: Reduce memory and query time
- **Resolution Scaling**: Lower resolution for speed
- **Point Sampling**: Adjust `point_step` for density

### Memory Management
- Automatic downsampling at `max_points`
- Spatial hashing for efficient storage
- Bounded queues prevent unbounded growth
- Periodic saves reduce memory pressure

---

## Extensibility

### Adding New Depth Models

```python
from eridian import ModelRegistry, BaseDepthEstimator

class CustomEstimator(BaseDepthEstimator):
    def _load_model(self):
        # Load custom model
        pass
    
    def _get_transform(self):
        # Return transform
        pass

# Register
ModelRegistry.register(
    ModelSpec(
        name="custom_model",
        display_name="Custom Model",
        hub_source="your/repo",
        model_name="model_name",
        transform_name="transforms",
    ),
    CustomEstimator
)
```

### Adding Pipeline Stages

```python
from eridian import AsyncPipeline, PipelineStage

pipeline = AsyncPipeline()

def custom_stage(frame_data):
    # Process frame_data
    frame_data.custom_result = ...  # Add custom data
    return frame_data

pipeline.register_stage(PipelineStage.VISUALIZE, custom_stage)
```

### Custom Spatial Indexing

```python
from eridian import BaseDepthEstimator

class CustomIndex:
    def insert(self, position, metadata):
        pass
    
    def query_radius(self, position, radius):
        pass
    
    # ... implement other methods
```

---

## Configuration Mapping

### Configuration → Components

```yaml
camera:        → CameraManager
depth:         → DepthEstimator
splat:         → SplatBuilder
performance:   → AsyncPipeline, SplatManager
slam:          → PoseTracker
logging:       → EridianLogger
```

### Runtime Overrides

Components can override config values:
- `CameraManager` adjusts resolution based on hardware
- `DepthEstimator` selects model based on GPU memory
- `AsyncPipeline` adjusts queue sizes based on throughput

---

## Error Handling Strategy

### Error Categories
1. **Recoverable**: Retry with backoff
2. **Non-critical**: Log and continue
3. **Critical**: Stop pipeline, save state

### Handling Flow
```
Error occurs
    ↓
ErrorHandler.handle_error()
    ↓
Log error + update counter
    ↓
Check retry limit
    ↓
Exceeded? → Stop / Return default
    ↓
Continue operation
```

### Crash Resilience
- Atomic file writes prevent corruption
- Periodic saves preserve state
- Graceful degradation on errors
- State preserved in SplatBuilder

---

## Testing Strategy

### Unit Tests
- Individual component functionality
- Mock external dependencies
- Edge cases and error conditions

### Integration Tests
- Component interactions
- End-to-end pipelines
- Real camera input (optional)

### Performance Tests
- Benchmarking suite
- Memory usage profiling
- FPS measurement under load

---

## Future Enhancements

### Phase 1 (Current)
- ✅ Modular architecture
- ✅ Multi-model depth
- ✅ Async pipeline
- ✅ Spatial indexing

### Phase 2 (Planned)
- GPU acceleration for SLAM
- Loop closure integration
- Better scale estimation
- Real-time mesh extraction

### Phase 3 (Future)
- Multi-camera support
- LiDAR integration
- Cloud processing
- Collaborative mapping

---

## Dependencies

### Core Dependencies
- Python 3.8+
- NumPy
- OpenCV
- PyTorch
- PyYAML
- Rerun SDK

### Optional Dependencies
- CUDA Toolkit (NVIDIA)
- Metal (Apple Silicon)
- Additional depth models

### System Requirements
- macOS (Apple Silicon preferred) or Linux
- 8GB+ RAM
- GPU recommended but not required

---

## Security Considerations

### Privacy
- All processing is local
- No cloud dependencies by default
- Optional cloud upload for backup

### Data Integrity
- Atomic file writes
- Checksum verification (future)
- Recovery from corrupted files (future)

### Resource Limits
- Configurable memory limits
- Bounded queues
- Graceful degradation

---

## Documentation

### User Documentation
- README.md: Quick start guide
- docs/API.md: Complete API reference
- docs/ARCHITECTURE.md: This document

### Developer Documentation
- Inline docstrings
- Type hints throughout
- Example code in __main__ sections

### Contribution Guide (Future)
- Code style guidelines
- Pull request process
- Issue templates

---

## Performance Benchmarks

### Typical Performance (Apple M2)

| Configuration | FPS | Memory | Quality |
|---------------|-----|--------|---------|
| Small Model   | 15-20| 2 GB   | Good    |
| Hybrid Model  | 8-12 | 3 GB   | Better  |
| Large Model   | 4-6  | 4 GB   | Best    |

### Scaling Characteristics

- **FPS**: Inversely proportional to resolution
- **Memory**: Linear with max_points
- **Quality**: Proportional to depth model accuracy

---

## Glossary

- **Gaussian Splat**: 3D representation using Gaussian distributions
- **SLAM**: Simultaneous Localization and Mapping
- **Visual Odometry**: Camera motion estimation from video
- **Monocular Depth**: Depth estimation from single camera
- **Spatial Hash**: Grid-based spatial indexing
- **Octree**: Hierarchical tree-based spatial structure
- **PLY**: Polygon File Format (3DGS variant)

---

## Contact & Support

- GitHub: https://github.com/Eeman1113/Eridian
- Issues: Report bugs and feature requests
- Documentation: See docs/ directory

---

**Last Updated**: 2025-04-05
**Version**: 0.1.0