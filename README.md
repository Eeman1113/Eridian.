# Eridian

**Real-time 3D world reconstruction from a single camera.**

Eridian turns any webcam into a spatial scanner. It watches what you see, understands how far away everything is, tracks how you move, and builds a 3D colored map of your surroundings — all in real time, on a laptop, with no special hardware.

[![PyPI](https://img.shields.io/pypi/v/eridian)](https://pypi.org/project/eridian/)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![Platform](https://img.shields.io/badge/platform-macOS%20%7C%20Linux-lightgrey)
![License](https://img.shields.io/badge/license-MIT-green)

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

## Install

### Option 1: pip install (recommended)

```bash
pip install eridian
eridian                # launch with webcam
eridian --test         # run on test video
eridian --video v.mp4  # any video file
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

### Use as a library

```python
from eridian import DepthEstimator, PointCloud, PoseEstimator

depth_est = DepthEstimator()
depth_map = depth_est.estimate(frame)
```

### Test mode (no camera needed)

```bash
eridian --test                         # process data/video.mp4 headless
eridian --video path/to/vid.mp4        # any video file
python render_video.py                 # render 4-panel demo video
```

If no camera is detected, Eridian automatically falls back to `data/video.mp4`.

## Requirements

- Python 3.10 – 3.13
- A webcam (built-in, USB, or macOS Continuity Camera) — or a video file for test mode
- CPU-only — no CUDA needed (uses Apple MPS when available)

## Controls

| Key | Action |
|-----|--------|
| `q` | Quit (in any OpenCV window) |
| `Ctrl+C` | Graceful shutdown with final save |
| Mouse | Orbit / zoom / pan in 3D window |

## Output files

| Path | What |
|------|------|
| `splat/cloud_latest.ply` | Latest point cloud (saved every 10s) |
| `splat/cloud_YYYYMMDD_HHMMSS.ply` | Timestamped backups (every 60s) |
| `splat/cloud_final_*.ply` | Final save on shutdown |
| `splat/depth_frames/*.png` | 16-bit metric depth maps (every 5th frame) |
| `output_video/eridian_demo.mp4` | 4-panel demo video |
| `logs/mapper.log` | Full application log |

## Architecture

Single `main.py`, no external config, no separate processes:

```
CameraCapture        — OpenCV with auto-reconnect + video fallback
DepthEstimator       — Depth Anything V2 Metric + bilateral filter + temporal EMA
PoseEstimator        — GFTT corners + LK optical flow + PnP (solvePnPRansac)
PointCloud           — Edge/normal filtering + keyframe gating + voxel averaging
Visualizer3D         — PyVista non-blocking renderer
save_ply()           — Binary PLY writer
WorldMapper          — Main loop with keyframe system
```

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
