# Environment Mapping Guide

Build a real 3D map of a physical space using Eridian's LingBot-MAP (GCT) backend.

## Overview

The GCT backend uses a Geometric Context Transformer to reconstruct 3D geometry from a video. You record a video walking through a space, feed it to Eridian, and get back a dense colored 3D point cloud of the environment.

Unlike the classic pipeline (which runs in real-time but builds the map incrementally), GCT processes the full video at once and jointly reasons about camera poses and scene geometry across all frames. This produces more accurate and complete 3D maps.

## What You Need

- Eridian installed (`pip install eridian`)
- A GCT model checkpoint (download from [LingBot-MAP](https://github.com/Robbyant/lingbot-map))
- A video of the space you want to map
- ~4 GB VRAM (Apple MPS or NVIDIA CUDA)

## Step 1: Record Your Video

The quality of your 3D map depends heavily on the input video. Follow these guidelines:

### Movement
- **Walk slowly and steadily** — fast motion causes blur which degrades reconstruction
- **Smooth turns** — avoid sudden rotations
- **Overlap** — every new frame should share at least 60% of visible content with the previous frame
- **Cover the full space** — walk the perimeter, then through the center

### Camera
- Hold the camera at chest height, pointed forward
- Keep the camera roughly horizontal (avoid pointing at floor/ceiling unless you want to map them)
- 30-60 seconds is enough for a single room
- 2-3 minutes for a full apartment or office floor

### Lighting
- Even, diffuse lighting works best
- Avoid pointing directly at bright windows or lights
- Avoid very dark areas — the model needs visible texture

### What Works Well
- Indoor rooms with furniture and wall features
- Hallways and corridors
- Outdoor paths and building exteriors
- Any space with visible texture and geometry

### What to Avoid
- Featureless white walls (no texture to track)
- Mirrors and glass (confuse depth estimation)
- Moving people/objects in the scene
- Extremely dark or extremely bright scenes

## Step 2: Run GCT Reconstruction

### Basic usage

```bash
eridian --gct --gct-model gct_checkpoint.pt --video room_scan.mp4
```

This will:
1. Load the GCT transformer model
2. Sample frames from your video at 10 FPS
3. Run streaming inference (joint pose + depth + 3D prediction)
4. Filter points by confidence
5. Save the result to `splat/gct_cloud.ply`

### Save to a specific location

```bash
eridian --gct --gct-model gct_checkpoint.pt --video room_scan.mp4 --savedir ~/3d_maps/
```

### Higher quality (denser point cloud)

```bash
eridian --gct --gct-model gct_checkpoint.pt --video room_scan.mp4 \
    --gct-conf 1.0 --gct-subsample 2
```

- `--gct-conf 1.0` — lower threshold = keep more points (default 1.5)
- `--gct-subsample 2` — less spatial downsampling = denser cloud (default 4)

### Faster processing (fewer frames)

```bash
eridian --gct --gct-model gct_checkpoint.pt --video room_scan.mp4 \
    --gct-fps 5 --gct-max-frames 100
```

### Long videos

For videos longer than ~2 minutes, the KV cache can grow large. Use a smaller sliding window:

```bash
eridian --gct --gct-model gct_checkpoint.pt --video long_walk.mp4 \
    --gct-window 32 --gct-fps 5
```

## Step 3: View Your 3D Map

The output `.ply` file is a standard point cloud format. Open it in:

- **[MeshLab](https://www.meshlab.net/)** — free, lightweight, good for quick viewing
- **[CloudCompare](https://www.danielgm.net/cc/)** — free, powerful measurement and analysis tools
- **Blender** — File > Import > PLY (good for rendering)
- **Open3D** (Python) — `o3d.io.read_point_cloud("gct_cloud.ply")`

## Step 4: Use in Your Own Code

```python
from eridian.gct_backend import GCTReconstructor, gct_to_pointcloud
from eridian import save_ply

# Load model
gct = GCTReconstructor(model_path="gct_checkpoint.pt")

# Reconstruct
result = gct.reconstruct_video("room_scan.mp4", fps=10)

# Extract point cloud
points, colors = gct_to_pointcloud(result, conf_threshold=1.5, subsample=4)
save_ply("room.ply", points, colors)

# Access raw outputs
depths = result["depth"]           # [N, H, W] metric depth maps
poses = result["extrinsic"]        # [N, 3, 4] camera extrinsics (c2w)
intrinsics = result["intrinsic"]   # [N, 3, 3] camera intrinsics
world_pts = result["world_points"] # [N, H, W, 3] per-pixel 3D coordinates
confidence = result["world_points_conf"]  # [N, H, W] per-pixel confidence
```

## Parameter Reference

| Parameter | Default | Effect |
|-----------|---------|--------|
| `--gct-fps` | 10 | Frames sampled per second from video. Lower = faster, fewer points |
| `--gct-max-frames` | all | Cap on total frames processed |
| `--gct-conf` | 1.5 | Minimum confidence to keep a point. Lower = more points, more noise |
| `--gct-subsample` | 4 | Spatial downsampling. 1 = all pixels, 4 = every 4th pixel |
| `--gct-scale-frames` | 8 | Initial frames used for scale calibration (bidirectional attention) |
| `--gct-window` | 64 | KV cache window size. Smaller = less memory, less temporal context |
| `--gct-image-size` | 518 | Input resolution. Higher = more detail, more compute |
| `--gct-keyframe-interval` | auto | How often to cache KV entries. Auto adjusts based on video length |

## Troubleshooting

### "No points passed confidence threshold"
Your confidence threshold is too high for the reconstruction quality. Try `--gct-conf 0.5` or even `--gct-conf 0.1`.

### Out of memory
- Reduce `--gct-window` (e.g., 32 or 16)
- Reduce `--gct-fps` (e.g., 5)
- Limit frames with `--gct-max-frames 100`
- Increase `--gct-subsample` (e.g., 8)

### Point cloud looks noisy
- Increase `--gct-conf` (e.g., 2.0 or 3.0) to keep only high-confidence points
- Check that your video has good lighting and isn't too shaky

### Point cloud has holes
- Record with more overlap between frames
- Lower `--gct-conf` to keep more points
- Use `--gct-subsample 1` for maximum density
