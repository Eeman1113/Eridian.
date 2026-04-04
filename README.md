# Eridian
### Map the world in 3D. One frame at a time.

  > Inspired by Rocky's tactile world model in Project Hail Mary —
  > building a real-time 3D map of your environment that could eventually
  > be converted into haptic feedback for the visually impaired.

---

## What This Does

```
iPhone Camera (Continuity)
        |
        v
 MiDaS Depth Network  -->  Per-pixel depth map (neural)
        |
        v
 Optical Flow Tracker  -->  Camera pose (R|t) in world space
        |
        v
 Gaussian Splat Builder  -->  Colored 3D gaussians in world coords
        |
        +-->  Rerun live viewer  (3D + depth + RGB panels, auto-opens)
        |
        +-->  ./splat/splat.ply  (saved every 30 frames, atomically)
```

Each pixel from the depth map becomes a 3D Gaussian with:

  - Position  -- back-projected from camera using depth + intrinsics
  - Color     -- sampled from the RGB frame
  - Scale     -- sized proportionally to depth and pixel footprint
  - Opacity   -- higher confidence for closer objects
  - Rotation  -- identity (future: align to surface normals)

---

## Quick Start

```bash
python3 run.py
```

First run creates a venv, installs everything, downloads MiDaS, and starts.
Every run after that launches instantly. The Rerun viewer opens automatically.

---

## Why Rerun instead of Open3D?

`open3d` has no Python 3.12/3.13 wheels and is effectively unmaintained on
Apple Silicon. Rerun is a modern replacement:

  - Supports Python 3.8 → 3.13 on macOS, Linux, Windows
  - Shows 3D point cloud + RGB feed + depth map in one tabbed viewer
  - Logs the camera trajectory automatically
  - Zero configuration — viewer spawns in a browser tab or native app

---

## Controls  (camera window focused)

```
  Q / Esc  --  Quit and save final splat
  S        --  Force save right now
  R        --  Reset map + pose (start fresh)
  D        --  Toggle depth visualization overlay in the capture window
  +        --  Denser point cloud (slower)
  -        --  Sparser point cloud (faster)
```

---

## Viewing Your Splat  (-.-)/

The .ply files saved to ./splat/ are in standard 3DGS format and open in:

```
  SuperSplat  -->  https://supersplat.playcanvas.com  (drag and drop)
  Luma AI     -->  https://lumalabs.ai/interactive-scenes
  nerfstudio  -->  ns-viewer --load-config ...
  Three.js    -->  THREE.GaussianSplatMesh
```

---

## Performance Tips  (^_^)

```
  Too slow (< 5 FPS)    -->  press - to reduce density,
                              or set POINT_STEP = 8 in run.py
  Better depth quality  -->  set DEPTH_MODEL = "DPT_Hybrid" (slower)
  GPU / M1 / M2 Mac     -->  auto-detected, check "[Depth] Device:" on startup
  Too many points       -->  lower MAX_POINTS in run.py
  Rerun viewer lag      -->  normal for large point clouds; the live PLY
                              is always up to date regardless
```

---

## How to Get Best Results

```
  (1)  Move slowly -- fast motion blurs depth estimates
  (2)  Overlap coverage -- revisit areas from different angles
  (3)  Good lighting -- depth networks need texture and contrast
  (4)  Avoid mirrors and glass -- they confuse depth networks
  (5)  Start from a fixed point -- origin is your first frame
```

---

## Architecture

### Depth Estimation (MiDaS)

  - Intel MiDaS model, runs fully local — no cloud, no API key
  - MiDaS_small  ~30ms per frame on CPU, ~5ms on GPU
  - DPT_Hybrid   better quality, ~80ms per frame on CPU
  - Output is inverse depth (disparity): 1 = close, 0 = far
  - Conversion:  metric_depth = 0.1 + (1 - disparity) * max_depth

### Visual Odometry

  - Lucas-Kanade sparse optical flow, 500 features tracked per frame
  - Essential matrix via RANSAC to get relative rotation + translation
  - Translation scale estimated from depth at tracked feature points

  [!]  Monocular VO drifts over time. No loop closure.
       Expect ~5–15 cm drift per metre travelled for room-scale scans.
       Press R if drift becomes obvious.

### Gaussian Splat Format  (3DGS .ply)

```
  x, y, z          -- position
  nx, ny, nz        -- normals (zero)
  f_dc_0/1/2        -- base color (SH coefficient, C0)
  opacity           -- logit-encoded
  scale_0/1/2       -- log-encoded anisotropic scale
  rot_0/1/2/3       -- quaternion rotation (wxyz)
```

### Save Strategy

  - Saves every 30 frames to splat/splat.ply via atomic file replace
    (write to .tmp then os.replace) — partial writes never corrupt the file
  - Timestamped backups every 150 frames:  splat/splat_000150.ply
  - Final save on quit, SIGINT, or camera disconnect

---

## The Rocky Vision  (>_<)

This is a foundation for what you described.
The path from here to a real tactile device:

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

## Honest Limitations  (-_-)

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

## Files

```
  run.py           -- everything in one file, self-installing
  README.md        -- this file
  splat/           -- output directory (created on first run)
    splat.ply            -- live file, updated every 30 frames
    splat_final.ply      -- written on clean exit
    splat_000150.ply     -- timestamped backups
```

---

```
                                              -- Rocky


                                                _____
                                               /     \
                                              | x _ x |
                                               \_____/
                                                  |
                                            ______|
                                           |      |
                                           |  ____|
                                           | |
                                           |_|
                                           ||||
                                           ||||
```
