# Python API Reference

## Classic API

### Eridian (high-level)

```python
from eridian import Eridian

e = Eridian()

# Depth from a single image
import cv2
frame = cv2.imread("photo.jpg")
depth = e.estimate_depth(frame)   # float32 HxW, values in meters

# Single frame → 3D points
points, colors = e.frame_to_points(frame)

# Process entire video
points, colors = e.process_video("scan.mp4")
e.save("output.ply")
```

### Streaming

```python
from eridian import Eridian

e = Eridian()
for result in e.stream(camera=0, max_frames=100):
    print(f"Frame {result.frame_index}: {e.point_count} points")

e.save("my_scan.ply")
```

**FrameResult fields:**

| Field | Type | Description |
|-------|------|-------------|
| `frame` | `np.ndarray` | BGR image |
| `depth` | `np.ndarray` | Float32 depth map (meters) |
| `pose` | `np.ndarray` | 4x4 camera pose matrix |
| `points` | `np.ndarray` | Accumulated Nx3 points |
| `colors` | `np.ndarray` | Accumulated Nx3 RGB colors |
| `is_keyframe` | `bool` | Whether this frame added geometry |
| `frame_index` | `int` | Frame number |

### Event Callbacks

```python
e = Eridian()
e.on("frame", lambda idx, frame, depth, pose: print(f"frame {idx}"))
e.on("keyframe", lambda idx, pts, cols: print(f"keyframe {idx}: {len(pts)} pts"))
e.on("depth", lambda idx, depth: print(f"depth: {depth.min():.1f}-{depth.max():.1f}m"))

e.process_video("scan.mp4")
```

### PLY I/O

```python
from eridian import save_ply, load_ply

save_ply("cloud.ply", points, colors)
points, colors = load_ply("cloud.ply")
```

### Camera Discovery

```python
from eridian import probe_cameras, pick_camera

cameras = probe_cameras()       # [{"index": int, "width": ..., "height": ...}]
chosen = pick_camera(cameras)   # interactive terminal picker
```

### Low-level Components

```python
from eridian import DepthEstimator, PoseEstimator, PointCloud

depth_est = DepthEstimator()
depth_map = depth_est.estimate(frame)

pose_est = PoseEstimator(fx, fy, cx, cy)
pose, keypoints, matches = pose_est.update(frame, depth=depth_map)

cloud = PointCloud()
cloud.add_frame(depth_map, frame, pose, K)
points, colors = cloud.get_data()
```

---

## GCT API (LingBot-MAP)

### GCTReconstructor

```python
from eridian.gct_backend import GCTReconstructor, gct_to_pointcloud

gct = GCTReconstructor(
    model_path="gct_checkpoint.pt",
    image_size=518,            # input resolution
    num_scale_frames=8,        # scale calibration frames
    conf_threshold=1.5,        # confidence filter
    kv_cache_sliding_window=64 # KV cache window
)
```

### Reconstruct from video

```python
result = gct.reconstruct_video(
    "room.mp4",
    fps=10,                    # frame sampling rate
    max_frames=None,           # process all frames
    keyframe_interval=1        # cache every frame
)
```

### Reconstruct from frame list

```python
frames = [cv2.imread(f) for f in sorted(glob("frames/*.jpg"))]
result = gct.reconstruct_frames(frames, keyframe_interval=1)
```

### Result dictionary

| Key | Shape | Description |
|-----|-------|-------------|
| `world_points` | `[N, H, W, 3]` | Per-pixel 3D world coordinates |
| `world_points_conf` | `[N, H, W]` | Per-pixel confidence |
| `depth` | `[N, H, W]` | Metric depth maps |
| `depth_conf` | `[N, H, W]` | Depth confidence |
| `extrinsic` | `[N, 3, 4]` | Camera-to-world transforms |
| `intrinsic` | `[N, 3, 3]` | Camera intrinsic matrices |
| `images` | `[N, 3, H, W]` | Preprocessed images [0,1] |
| `num_frames` | `int` | Number of frames processed |

### Extract point cloud

```python
points, colors = gct_to_pointcloud(
    result,
    conf_threshold=1.5,   # min confidence to keep
    subsample=4            # spatial downsampling (1=all, 4=every 4th)
)

save_ply("map.ply", points, colors)
```
