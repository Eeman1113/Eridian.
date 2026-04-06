# Eridian

Real-time monocular 3D world mapping from a single webcam. Eridian combines deep learning depth estimation with visual odometry to reconstruct a colored 3D point cloud of your environment as you move the camera around.

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![Platform](https://img.shields.io/badge/platform-macOS%20%7C%20Linux-lightgrey)
![License](https://img.shields.io/badge/license-MIT-green)

## How it works

1. **Captures** frames from your webcam (640x480 @ 30fps)
2. **Estimates depth** per-pixel using [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) (small model)
3. **Tracks camera motion** via ORB feature matching + Essential Matrix decomposition
4. **Back-projects** pixels to 3D using estimated intrinsics + accumulated pose
5. **Builds** a global colored point cloud with voxel downsampling
6. **Visualizes** the 3D map in real time (PyVista) alongside 2D debug views (OpenCV)
7. **Saves** PLY files automatically — viewable in MeshLab, CloudCompare, or Blender

## Quick start

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
python test_components.py   # verify depth model + PLY export
python main.py              # launch
```

## Requirements

- Python 3.10 – 3.13
- A webcam (built-in, USB, or macOS Continuity Camera)
- CPU-only — no CUDA needed (uses Apple MPS when available)

## Controls

| Key | Action |
|-----|--------|
| `q` | Quit (in any OpenCV window) |
| `Ctrl+C` | Graceful shutdown with final save |
| Mouse | Orbit / zoom / pan in 3D window |

## What you see

Eridian opens four windows:

- **Camera** — live feed with FPS and point count overlay
- **Depth** — colorized depth map (INFERNO colormap)
- **Features** — ORB keypoints and match lines
- **3D World Map** — accumulated point cloud you can orbit around

## Output files

| Path | What |
|------|------|
| `splat/cloud_latest.ply` | Latest point cloud (saved every 10s) |
| `splat/cloud_YYYYMMDD_HHMMSS.ply` | Timestamped backups (every 60s) |
| `splat/cloud_final_*.ply` | Final save on shutdown |
| `splat/depth_frames/*.png` | 16-bit depth maps (every 5th frame) |
| `logs/mapper.log` | Full application log |

## Architecture

Single `main.py`, no external config, no separate processes:

```
CameraCapture        — OpenCV with auto-reconnect
DepthEstimator       — Depth Anything V2 (HuggingFace transformers)
PoseEstimator        — ORB + Essential Matrix visual odometry
PointCloud           — Accumulation + vectorized voxel downsampling
Visualizer3D         — PyVista non-blocking renderer
save_ply()           — Binary PLY writer
WorldMapper          — Main loop tying it all together
```

## Depth model fallback chain

If Depth Anything V2 Small fails to load, Eridian tries these in order:

1. `depth-anything/Depth-Anything-V2-Small-hf`
2. `depth-anything/Depth-Anything-V2-Base-hf`
3. `Intel/dpt-swinv2-tiny-256`
4. `Intel/dpt-hybrid-midas`

## Performance

On Apple M-series (MPS):
- Depth inference: ~5–6 FPS
- ORB tracking: <2ms per frame
- Point cloud update: <5ms per frame
- 3D visualization: updates every 5 frames

## Viewing PLY files

The `.ply` files Eridian produces can be opened in:
- [MeshLab](https://www.meshlab.net/)
- [CloudCompare](https://www.danielgm.net/cc/)
- Blender (File → Import → PLY)
- Any viewer supporting binary little-endian PLY with vertex colors

## License

MIT
