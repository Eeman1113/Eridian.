# Classic Pipeline

The default Eridian backend uses a multi-stage pipeline to reconstruct 3D geometry from a camera stream in real-time.

## Pipeline Stages

### 1. Metric Depth Estimation

A neural network ([Depth Anything V2 Metric](https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf)) estimates the real-world distance in meters from the camera to every pixel. The depth is smoothed with a bilateral filter and temporally stabilized with exponential moving average to reduce flicker.

**Model fallback chain:**
1. `depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf` (metric)
2. `depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf` (metric)
3. `depth-anything/Depth-Anything-V2-Small-hf` (relative)
4. `Intel/dpt-swinv2-tiny-256` (relative)
5. `Intel/dpt-hybrid-midas` (relative)

### 2. Camera Motion Tracking

Eridian tracks hundreds of corner features across consecutive frames using Lucas-Kanade optical flow. Each tracked point is lifted into 3D using the depth map. These 3D-to-2D correspondences feed into a PnP (Perspective-n-Point) solver to compute the exact camera motion between frames.

A forward-backward consistency check eliminates bad tracks before they corrupt the pose.

### 3. Intelligent Point Cloud Accumulation

A keyframe system detects when the camera has moved enough (>8cm or >5 degrees) to add new geometry. Three quality filters run on each point:

- **Depth edge rejection** — Removes "flying pixels" at object boundaries (Sobel gradient detection)
- **Grazing angle rejection** — Removes points on surfaces viewed at >75 degrees
- **Voxel averaging** — All points within each 3cm voxel are averaged for cleaner surfaces

### 4. Live Visualization

The point cloud renders in real-time alongside camera feed, depth map, and feature tracking view using PyVista.

## Architecture

```
CameraCapture        — OpenCV with auto-reconnect + video fallback
DepthEstimator       — Depth Anything V2 Metric + bilateral filter + temporal EMA
PoseEstimator        — GFTT corners + LK optical flow + PnP (solvePnPRansac)
PointCloud           — Edge/normal filtering + keyframe gating + voxel averaging
Visualizer3D         — PyVista non-blocking renderer
save_ply()           — Binary PLY writer
WorldMapper          — Main loop with keyframe system
```

## Performance

On Apple M-series (MPS):
- Depth inference: ~5 FPS
- Optical flow tracking: <3ms per frame
- PnP pose solve: <1ms
- Point filtering + accumulation: ~5ms per frame
- **Total pipeline: ~5 FPS real-time**
