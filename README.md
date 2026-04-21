# Eridian

**Real-time 3D world reconstruction from a single camera.**

Eridian turns any webcam into a spatial scanner. It watches what you see, understands how far away everything is, tracks how you move, and builds a 3D colored map of your surroundings — all in real time, on a laptop, with no special hardware.

[![PyPI](https://img.shields.io/pypi/v/eridian)](https://pypi.org/project/eridian/)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![Platform](https://img.shields.io/badge/platform-macOS%20%7C%20Linux-lightgrey)
![License](https://img.shields.io/badge/license-MIT-green)
[![Docs](https://img.shields.io/badge/docs-Wiki-blue)](https://github.com/Eeman1113/Eridian./tree/main/docs)

---

## Demo

![Eridian 4-panel view](https://github.com/Eeman1113/Eridian./raw/main/assets/demo_4panel.jpg)

> **Top-left:** Live camera feed | **Top-right:** Metric depth map | **Bottom-left:** Optical flow tracking | **Bottom-right:** Accumulated 3D point cloud

https://github.com/Eeman1113/Eridian./raw/main/output_video/eridian_demo.mp4

---

## What it does

Eridian takes a flat 2D video stream and reconstructs the 3D structure of the world from it. Every frame goes through four stages:

### 1. Metric Depth Estimation

![Depth map](https://github.com/Eeman1113/Eridian./raw/main/assets/panel_depth.jpg)

A neural network ([Depth Anything V2 Metric](https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf)) estimates the real-world distance in meters from the camera to every single pixel in the frame. This isn't relative "closer vs farther" — it outputs actual metric depth (e.g., "this wall is 2.3 meters away"). The depth is then smoothed with a bilateral filter to reduce noise while keeping sharp edges, and temporally stabilized so it doesn't flicker between frames.

### 2. Camera Motion Tracking

![Optical flow](https://github.com/Eeman1113/Eridian./raw/main/assets/panel_features.jpg)

Eridian tracks hundreds of corner features across consecutive frames using Lucas-Kanade optical flow. Each tracked point is lifted into 3D using the depth map, creating a set of known 3D-to-2D correspondences. These are fed into a PnP (Perspective-n-Point) solver that computes exactly how the camera moved between frames — both direction and distance, in real meters. A forward-backward consistency check eliminates bad tracks before they can corrupt the pose.

### 3. Intelligent Point Cloud Accumulation

![3D Point Cloud](https://github.com/Eeman1113/Eridian./raw/main/assets/panel_pointcloud.jpg)

Not every frame contributes to the 3D map. A keyframe system detects when the camera has moved enough (>8cm or >5 degrees) to justify adding new geometry. When a keyframe fires, each pixel is back-projected from 2D into 3D world coordinates using the depth and the accumulated camera pose. Three quality filters run before any point is accepted:

- **Depth edge rejection** — Removes "flying pixels" at object boundaries where depth is unreliable (detected via Sobel gradients)
- **Grazing angle rejection** — Removes points on surfaces viewed at steep angles (>75 degrees), where depth accuracy degrades
- **Voxel averaging** — Instead of keeping random points, all points within each 3cm voxel are averaged together, producing cleaner surfaces

### 4. Live 3D Visualization

The accumulated point cloud is rendered in real time alongside the camera feed, depth map, and feature tracking view. The result is a colored 3D map that grows as you move the camera around.

---

## Why "Eridian"?

This project is named after Rocky, the Eridian engineer from Andy Weir's *Project Hail Mary*.

Rocky is blind. His entire species is. They have no concept of vision. They understand the world through sound and touch. When Rocky needs to communicate spatial concepts with a human, he builds a device: a flat surface covered with a grid of metal pins driven by tiny motors. Each pin rises and falls independently. Run your hand across it and you feel the shape of a star map, a molecule, a room. It is a screen for someone who cannot see.

I read that and couldn't stop thinking about it.

**700 million people on Earth live with significant vision impairment.** They navigate the world with canes, memorized routes, and other people's descriptions. Understanding the 3D layout of an unfamiliar room, where the furniture is, how far the doorway is, whether there's a step ahead, requires either experience or help.

Rocky's pin device is fiction. But the idea isn't.

## The endgame: a real-world Rocky device

The long-term goal of this project is to build a physical tactile display, a surface of servo-driven pins that conforms in real time to whatever a phone camera is looking at.

Point your phone at a room. The pin surface reshapes itself into a miniature relief of that room. Feel the couch on the left, the table ahead, the doorway to the right. Move the phone and the surface updates. It's a live tactile map of the world around you.

**How it works (the vision):**

1. **Eridian runs on the phone** and reconstructs the 3D scene from the camera feed, exactly as it does now
2. **The 3D point cloud gets downsampled** to match the resolution of the physical pin grid (say, 32x32 or 64x64 pins)
3. **Each pin's height maps to the depth** at that grid position, driven by a small servo or linear actuator
4. **The surface updates in real time** as the phone scans the environment, reshaping itself to match whatever the camera sees

This is not a screen. It's not audio. It's direct spatial understanding through touch, the same way Rocky's species understands the universe.

## Where Eridian fits today

Eridian is the perception layer. It solves the first and hardest problem: taking a flat 2D video from a single cheap camera and turning it into accurate 3D geometry, in real time, on consumer hardware.

The software side is the foundation everything else gets built on:

- **Obstacle detection and distance warnings** - "Table 1.5 meters ahead, slightly left." The metric depth map already provides this at every pixel, every frame.

- **Room layout narration** - Accumulate the 3D map over time, then describe the space: "Rectangular room, about 4 by 6 meters. Door behind you to the right. Couch along the left wall."

- **Path planning** - Analyze the point cloud for clear walking paths and floor-level obstacles that a cane might miss, like a low table or an open cabinet door at head height.

- **Spatial memory** - A persistent 3D map remembers the entire space. Scan a room once and the system knows what's there even when the camera isn't pointing at it.

- **Indoor navigation** - Combined with visual place recognition, accumulated 3D maps could enable turn-by-turn navigation inside buildings where GPS doesn't work.

**Why single-camera matters:** Existing spatial sensing (LiDAR devices) is expensive and locked to specific hardware. Eridian runs on any camera, including the phone in someone's pocket. The hardware barrier drops to zero.

**The next steps are:** the servo-driven pin surface prototype, the phone-to-device communication layer, and the haptic feedback design. Eridian is the eyes. The pin grid is the hands. Together they give spatial understanding to someone who has neither.

**Eridian is the perception layer. The pin grid is the interface. Rocky showed us the idea. We're building it for real.**

---

## Pipeline at a glance

![Early in the scan](https://github.com/Eeman1113/Eridian./raw/main/assets/demo_early.jpg)
*Early in the scan — depth map is active, point cloud is starting to form*

![Full reconstruction](https://github.com/Eeman1113/Eridian./raw/main/assets/demo_late.jpg)
*After scanning — dense point cloud with room geometry visible*

---

## Two Backends

Eridian ships with two reconstruction backends — pick the one that fits your use case:

| | **Classic** (default) | **LingBot-MAP (GCT)** |
|---|---|---|
| **How it works** | Depth Anything V2 + optical flow + PnP | Single transformer predicts pose, depth, and 3D points jointly |
| **Input** | Webcam or video | Video file (camera streaming coming soon) |
| **Real-time** | Yes (~5 FPS on Apple M-series) | Not yet — batch processes video |
| **Hardware** | CPU / MPS, no GPU required | MPS / CUDA, ~4 GB VRAM |
| **Best for** | Live scanning, interactive use | High-quality offline 3D maps of environments |
| **Output** | Colored PLY point cloud | Colored PLY point cloud |

### When to use LingBot-MAP

The GCT (Geometric Context Transformer) backend from [LingBot-MAP](https://github.com/Robbyant/lingbot-map) replaces the entire classic pipeline with a single transformer model. Instead of running separate depth estimation, feature tracking, and pose solving steps, GCT processes all frames through one network that jointly understands camera motion and scene geometry.

This makes it significantly better for **mapping a full environment** — walk through a room, hallway, or outdoor space with your phone camera, then feed the video to Eridian GCT to get a dense, accurate 3D map.

---

## Install

### Option 1: pip install (recommended)

```bash
pip install eridian
eridian                    # launch with webcam (interactive camera picker)
eridian --camera 0         # use specific camera index
eridian --test             # run on test video
eridian --video v.mp4      # any video file
eridian --save             # save PLY to current directory on exit
eridian --savedir ~/Downloads    # save PLY files to a custom directory
eridian --render --video v.mp4   # render 4-panel demo video
```

### Option 2: clone and run

```bash
git clone https://github.com/Eeman1113/Eridian..git
cd Eridian.
./run.sh
```

`run.sh` handles everything: creates a virtualenv, installs dependencies, runs component tests, and launches the mapper.

### Manual setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
python test_components.py   # verify depth model + PLY export
python main.py              # launch
```

### Test mode (no camera needed)

```bash
eridian --test                         # process data/video.mp4 headless
eridian --video path/to/vid.mp4        # any video file
eridian --render --video v.mp4         # render 4-panel demo video
```

If no camera is detected, Eridian automatically falls back to `data/video.mp4`.

---

## Python API

Eridian exposes a full Python API for integration into your own projects.

### Quick start

```python
from eridian import Eridian

e = Eridian()

# Estimate depth from a single image
import cv2
frame = cv2.imread("photo.jpg")
depth = e.estimate_depth(frame)   # float32 HxW, values in meters

# Convert a frame directly to 3D points
points, colors = e.frame_to_points(frame)

# Process an entire video
points, colors = e.process_video("scan.mp4")
e.save("output.ply")
```

### Stream from camera

```python
from eridian import Eridian

e = Eridian()
for result in e.stream(camera=0, max_frames=100):
    print(f"Frame {result.frame_index}: {e.point_count} points, keyframe={result.is_keyframe}")

e.save("my_scan.ply")
```

`stream()` is a generator that yields `FrameResult` objects:

| Field | Type | Description |
|-------|------|-------------|
| `frame` | `np.ndarray` | BGR image |
| `depth` | `np.ndarray` | Float32 depth map (meters) |
| `pose` | `np.ndarray` | 4x4 camera pose matrix |
| `points` | `np.ndarray` | Accumulated Nx3 points |
| `colors` | `np.ndarray` | Accumulated Nx3 RGB colors |
| `is_keyframe` | `bool` | Whether this frame added geometry |
| `frame_index` | `int` | Frame number |

### Event callbacks

```python
e = Eridian()
e.on("frame", lambda idx, frame, depth, pose: print(f"frame {idx}"))
e.on("keyframe", lambda idx, pts, cols: print(f"keyframe {idx}: {len(pts)} pts"))
e.on("depth", lambda idx, depth: print(f"depth range: {depth.min():.1f}-{depth.max():.1f}m"))

e.process_video("scan.mp4")
```

### Save and load point clouds

```python
from eridian import Eridian, save_ply, load_ply

e = Eridian()
e.process_video("scan.mp4")
e.save("scan.ply")

# Load back
e.load("scan.ply")
print(e.point_count)

# Or use standalone functions
points, colors = load_ply("scan.ply")
save_ply("copy.ply", points, colors)
```

### Camera discovery

```python
from eridian import probe_cameras, pick_camera

cameras = probe_cameras()         # list of {"index": int, "width": ..., "height": ...}
chosen = pick_camera(cameras)     # interactive terminal picker, returns index
```

### Low-level components

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

## Requirements

- Python 3.10 – 3.13
- A webcam (built-in, USB, or macOS Continuity Camera) — or a video file for test mode
- CPU-only — no CUDA needed (uses Apple MPS when available)

## 3D Environment Mapping with LingBot-MAP

This is the main workflow for building a real 3D map of a physical space.

### Step 1: Record a video

Walk slowly through the space you want to map. Use your phone or any camera. Tips:
- **Move slowly and steadily** — fast motion causes blur
- **Overlap** — make sure consecutive frames share visible geometry
- **Cover the space** — walk the full perimeter and through the center
- **Lighting** — even lighting works best, avoid pointing at bright windows
- 30-60 seconds of video is enough for a room; longer for larger spaces

### Step 2: Get the GCT checkpoint

Download the LingBot-MAP model weights from [the official repo](https://github.com/Robbyant/lingbot-map):

```bash
# Example — check the LingBot-MAP repo for the latest checkpoint URL
wget -O gct_checkpoint.pt <checkpoint-url>
```

### Step 3: Run Eridian in GCT mode

```bash
# Basic — reconstruct from video
eridian --gct --gct-model gct_checkpoint.pt --video room_scan.mp4

# Save output to a specific directory
eridian --gct --gct-model gct_checkpoint.pt --video room_scan.mp4 --savedir ~/3d_maps/

# Higher quality — keep more points, less subsampling
eridian --gct --gct-model gct_checkpoint.pt --video room_scan.mp4 \
    --gct-conf 1.0 --gct-subsample 2

# Faster — process fewer frames
eridian --gct --gct-model gct_checkpoint.pt --video room_scan.mp4 \
    --gct-fps 5 --gct-max-frames 100
```

### Step 4: View your 3D map

The output is a `.ply` point cloud file. Open it in:
- [MeshLab](https://www.meshlab.net/) — free, lightweight
- [CloudCompare](https://www.danielgm.net/cc/) — free, powerful measurement tools
- Blender — File > Import > PLY

### Python API (GCT)

```python
from eridian.gct_backend import GCTReconstructor, gct_to_pointcloud
from eridian import save_ply

# Load model
gct = GCTReconstructor(model_path="gct_checkpoint.pt")

# Reconstruct from video
result = gct.reconstruct_video("room_scan.mp4", fps=10)

# Extract colored point cloud
points, colors = gct_to_pointcloud(result, conf_threshold=1.5, subsample=4)

# Save
save_ply("room_map.ply", points, colors)
print(f"{len(points):,} points")

# Access per-frame data
print(f"Frames processed: {result['num_frames']}")
print(f"Depth maps shape: {result['depth'].shape}")        # [N, H, W]
print(f"Camera poses shape: {result['extrinsic'].shape}")   # [N, 3, 4]
print(f"World points shape: {result['world_points'].shape}")# [N, H, W, 3]
```

You can also reconstruct from a list of frames directly:

```python
import cv2

frames = [cv2.imread(f"frame_{i:04d}.jpg") for i in range(50)]
result = gct.reconstruct_frames(frames, keyframe_interval=1)
points, colors = gct_to_pointcloud(result)
save_ply("map.ply", points, colors)
```

---

## CLI Reference

### Classic mode

| Flag | Description |
|------|-------------|
| `--test` | Use `data/video.mp4` instead of camera |
| `--video PATH` | Process any video file |
| `--camera N` | Use camera index N (skip interactive picker) |
| `--headless` | No GUI windows, process and save only |
| `--save` | Save PLY to current working directory on exit |
| `--savedir DIR` | Save PLY files to a custom directory (default: `./splat/`) |
| `--render` | Render a 4-panel demo video from input |
| `--output PATH` | Output path for rendered video |
| `--model small\|base\|large` | Depth model size (default: small) |

### GCT mode (LingBot-MAP)

| Flag | Description |
|------|-------------|
| `--gct` | Enable GCT transformer backend |
| `--gct-model PATH` | Path to GCT checkpoint (.pt file) |
| `--gct-fps N` | Frame sampling rate from video (default: 10) |
| `--gct-max-frames N` | Max frames to process |
| `--gct-scale-frames N` | Scale estimation frames (default: 8) |
| `--gct-window N` | KV cache sliding window size (default: 64) |
| `--gct-conf FLOAT` | Confidence threshold for points (default: 1.5) |
| `--gct-subsample N` | Spatial subsampling factor (default: 4, lower = denser) |
| `--gct-image-size N` | Input resolution (default: 518) |
| `--gct-keyframe-interval N` | KV cache keyframe interval (auto if unset) |

## Controls

| Key | Action |
|-----|--------|
| `q` | Quit (in any OpenCV window) |
| `Ctrl+C` | Graceful shutdown with final save |
| Mouse | Orbit / zoom / pan in 3D window |

## Output files

### Classic mode
| Path | What |
|------|------|
| `splat/cloud_latest.ply` | Latest point cloud (saved every 10s, or `--savedir`) |
| `splat/cloud_YYYYMMDD_HHMMSS.ply` | Timestamped backups (every 60s, or `--savedir`) |
| `splat/cloud_final_*.ply` | Final save on shutdown (or `--savedir`) |
| `splat/depth_frames/*.png` | 16-bit metric depth maps (every 5th frame) |
| `output_video/eridian_demo.mp4` | 4-panel demo video |
| `logs/mapper.log` | Full application log |

### GCT mode
| Path | What |
|------|------|
| `splat/gct_cloud.ply` | Reconstructed 3D point cloud (or `--savedir`) |
| `eridian_gct_cloud.ply` | Copy in working directory (with `--save`) |

## Architecture

### Classic pipeline

```
CameraCapture        — OpenCV with auto-reconnect + video fallback
DepthEstimator       — Depth Anything V2 Metric + bilateral filter + temporal EMA
PoseEstimator        — GFTT corners + LK optical flow + PnP (solvePnPRansac)
PointCloud           — Edge/normal filtering + keyframe gating + voxel averaging
Visualizer3D         — PyVista non-blocking renderer
save_ply()           — Binary PLY writer
WorldMapper          — Main loop with keyframe system
```

### GCT pipeline (LingBot-MAP)

```
GCTReconstructor     — Loads and runs the GCT transformer model
 ├─ AggregatorStream — DINOv2 ViT backbone + causal temporal attention
 ├─ CameraHead       — Iterative camera pose prediction (translation + rotation + FOV)
 ├─ DPTHead          — Dense depth + 3D world point prediction with confidence
 └─ KV Cache         — Sliding window memory for streaming inference
gct_to_pointcloud()  — Confidence filtering + color extraction → PLY-ready arrays
```

The GCT model jointly predicts camera pose, metric depth, and 3D world coordinates in a single forward pass — no separate tracking or pose estimation needed.

## Depth model fallback chain

Eridian tries metric models first (real meters), then falls back to relative:

1. `depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf` (metric)
2. `depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf` (metric)
3. `depth-anything/Depth-Anything-V2-Small-hf` (relative)
4. `Intel/dpt-swinv2-tiny-256` (relative)
5. `Intel/dpt-hybrid-midas` (relative)

## Performance

On Apple M-series (MPS):
- Depth inference: ~5 FPS
- Optical flow tracking: <3ms per frame
- PnP pose solve: <1ms
- Point filtering + accumulation: ~5ms per frame
- Total pipeline: ~5 FPS real-time

## Viewing PLY files

The `.ply` files Eridian produces can be opened in:
- [MeshLab](https://www.meshlab.net/)
- [CloudCompare](https://www.danielgm.net/cc/)
- Blender (File > Import > PLY)
- Any viewer supporting binary little-endian PLY with vertex colors

## License

MIT
