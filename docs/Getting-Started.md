# Getting Started

## Install

```bash
pip install eridian
```

Requires Python 3.10+. Works on macOS (MPS) and Linux. No CUDA required for the classic pipeline.

## Quick Start — Classic (Live Scanning)

```bash
# Launch with webcam
eridian

# Use a specific camera
eridian --camera 0

# Process a video file
eridian --video walkthrough.mp4

# Save the point cloud
eridian --save
```

Move your camera around slowly. You'll see a live 4-panel view: camera feed, depth map, feature tracking, and growing 3D point cloud.

Press `q` to quit. The point cloud is auto-saved to `splat/`.

## Quick Start — GCT (Environment Mapping)

For higher-quality offline reconstruction of a full space:

```bash
# 1. Get a GCT checkpoint from https://github.com/Robbyant/lingbot-map

# 2. Run reconstruction
eridian --gct --gct-model gct_checkpoint.pt --video room_scan.mp4

# 3. Open splat/gct_cloud.ply in MeshLab, CloudCompare, or Blender
```

See [[Environment Mapping Guide]] for the full walkthrough.

## Which Backend Should I Use?

**Use Classic when:**
- You want real-time feedback while scanning
- You're streaming from a webcam
- You want quick results
- You don't have a GPU / large VRAM

**Use GCT (LingBot-MAP) when:**
- You want the most accurate 3D map possible
- You have a pre-recorded video of a space
- You need dense, complete coverage
- You have ~4 GB VRAM available (MPS or CUDA)

## Next Steps

- [[Environment Mapping Guide]] — Detailed guide for mapping rooms and spaces
- [[Python API Reference]] — Integrate Eridian into your code
- [[Tuning and Tips]] — Optimize your results
