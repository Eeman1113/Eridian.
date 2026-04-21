# LingBot-MAP (GCT) Backend

Technical deep-dive into Eridian's transformer-based 3D reconstruction backend.

**Credit:** [LingBot-MAP](https://github.com/Robbyant/lingbot-map) by the Robbyant Team, Apache 2.0 License.

## What is GCT?

GCT (Geometric Context Transformer) is a feed-forward transformer model that takes a sequence of images and jointly predicts:
- **Camera poses** — where the camera was for each frame (position + rotation + field of view)
- **Metric depth maps** — real-world distance from camera to every pixel
- **3D world points** — per-pixel 3D coordinates in a global reference frame
- **Confidence scores** — per-pixel reliability estimate

This replaces the classic Eridian pipeline (Depth Anything V2 + optical flow + PnP) with a single model that reasons about geometry and motion together.

## Architecture

```
Input frames [N, 3, 518, 518]
       │
       ▼
┌──────────────────────────┐
│   DINOv2 ViT-L/14        │  Patch embedding → 37x37 patch tokens per frame
│   (frozen backbone)       │
└──────────┬───────────────┘
           ▼
┌──────────────────────────┐
│   AggregatorStream        │  Causal temporal attention across frames
│   - Frame attention       │  Each frame attends to itself
│   - Global attention      │  Cross-frame temporal reasoning
│   - Special tokens:       │  Camera token, register tokens, scale token
│   - KV cache              │  Sliding window for memory efficiency
└──────────┬───────────────┘
           ▼
     ┌─────┴─────┐
     ▼           ▼
┌─────────┐ ┌──────────┐
│CameraHead│ │ DPT Head │
│9D pose:  │ │depth [H,W]│
│T(3)+Q(4) │ │pts [H,W,3]│
│+FOV(2)   │ │conf [H,W] │
└─────────┘ └──────────┘
```

### Key components

**AggregatorStream** — The core temporal reasoning module. Uses DINOv2 ViT-L/14 for patch embedding, then alternates between:
- *Frame attention*: self-attention within each frame's patches
- *Global attention*: causal cross-frame attention (each frame attends to all past frames)

A FlashInfer-based KV cache stores past frame representations, with a sliding window to bound memory usage. Falls back to SDPA (scaled dot-product attention) on non-CUDA devices.

**CameraHead** — Extracts a special camera token from each frame's representation and iteratively refines it through 4 attention blocks. Predicts a 9D vector: `[translation(3), quaternion(4), fov(2)]`.

**DPTHead** — Dense Prediction Transformer head (inspired by Depth Anything V2's architecture). Fuses multi-scale features from different aggregator blocks. Predicts per-pixel depth, 3D world coordinates, and confidence scores.

## Streaming Inference

GCT processes video in a streaming fashion:

1. **Scale frames** (first N frames, default 8): Processed together with bidirectional attention. This calibrates the model's understanding of scale and scene geometry.

2. **Streaming frames** (remaining): Processed one at a time with causal attention. Each frame attends to the KV cache (past frames) plus its own tokens. New KV entries are stored in the cache.

3. **Keyframe system**: Not every frame needs to store KV entries. The `keyframe_interval` parameter controls how often cache entries are added, reducing memory for long videos.

```python
# The inference loop (simplified)
model.inference_streaming(
    images,                    # [1, N, 3, H, W]
    num_scale_frames=8,        # First 8 frames = bidirectional
    keyframe_interval=1,       # Cache every frame
    output_device="cpu",       # Offload results to CPU per-frame
)
```

## 3D Reconstruction Pipeline

For each frame, GCT outputs:
- `world_points [H, W, 3]` — 3D coordinates of every pixel in world space
- `world_points_conf [H, W]` — confidence score per pixel
- `depth [H, W]` — metric depth map
- `pose_enc [9]` — camera pose encoding

The `gct_to_pointcloud()` function in `eridian/gct_backend.py`:
1. Iterates over all frames
2. Spatially subsamples points (default: every 4th pixel)
3. Filters by confidence threshold (default: 1.5)
4. Extracts RGB colors from the original image
5. Returns concatenated `(points, colors)` arrays ready for PLY export

## Pose Representation

The 9D pose encoding:
```
[0:3]   Translation (x, y, z) in meters — camera position in world frame
[3:7]   Rotation quaternion (x, y, z, w) — scalar-last convention
[7:9]   Field of view (horizontal, vertical) in radians
```

Converted to standard camera matrices via `pose_encoding_to_extri_intri()`:
- **Extrinsic** [3x4]: world-to-camera rotation and translation
- **Intrinsic** [3x3]: focal length and principal point from FOV

The extrinsic is then inverted (w2c → c2w) for world-frame point cloud generation.

## Memory and Performance

| Setting | Memory Impact | Quality Impact |
|---------|---------------|----------------|
| `gct-window 64` (default) | ~4 GB | Full temporal context |
| `gct-window 32` | ~2.5 GB | Good for most scenes |
| `gct-window 16` | ~1.5 GB | May miss long-range geometry |
| `gct-subsample 4` (default) | Low | 1/16th of pixels |
| `gct-subsample 1` | 16x more points | Full resolution |
| `gct-scale-frames 8` (default) | Minimal | Good calibration |
| `gct-image-size 518` (default) | Standard | 37x37 patch grid |

## Files

```
eridian/
├── gct_backend.py                 # GCTReconstructor wrapper + gct_to_pointcloud()
└── lingbot_map/
    ├── models/
    │   ├── gct_base.py            # Base model class
    │   ├── gct_stream.py          # Streaming inference with KV cache
    │   └── gct_stream_window.py   # Windowed streaming variant
    ├── aggregator/
    │   ├── base.py                # DINOv2 feature extractor
    │   └── stream.py              # Causal temporal aggregator
    ├── heads/
    │   ├── camera_head.py         # Camera pose prediction
    │   ├── dpt_head.py            # Dense depth + point prediction
    │   └── head_act.py            # Activation functions
    ├── layers/                    # Transformer building blocks
    │   ├── attention.py, block.py, rope.py, ...
    │   └── vision_transformer.py  # DINOv2 ViT variants
    ├── utils/
    │   ├── geometry.py            # 3D projections, SE(3) math
    │   ├── pose_enc.py            # Pose encoding/decoding
    │   └── rotation.py            # Quaternion utilities
    └── vis/
        ├── point_cloud_viewer.py  # Interactive Viser-based viewer
        ├── glb_export.py          # Export to GLB format
        └── sky_segmentation.py    # Sky region filtering
```
