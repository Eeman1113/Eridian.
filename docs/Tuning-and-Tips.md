# Tuning and Tips

## Recording Tips

### For best results
- Walk at a slow, steady pace (roughly 0.5 m/s)
- Keep the camera pointed roughly horizontally
- Make smooth turns — avoid jerky rotation
- Ensure consecutive frames overlap by at least 60%
- Good, even lighting matters more than camera quality

### Avoid
- Fast motion (causes blur, breaks tracking)
- Featureless surfaces (plain white walls)
- Mirrors and glass (confuse depth estimation)
- Moving objects in the scene
- Pointing at bright light sources

## Classic Pipeline Tuning

### Model size
```bash
eridian --model small   # fastest, ~5 FPS (default)
eridian --model base    # balanced
eridian --model large   # best depth quality, slower
```

### Scan technique
- Start by pointing at a textured area (bookshelf, furniture)
- Move slowly sideways before rotating
- Return to previously seen areas to close loops
- The keyframe system auto-detects when you've moved enough (>8cm or >5 degrees)

## GCT Tuning

### Quality vs Speed

**Maximum quality:**
```bash
eridian --gct --gct-model ckpt.pt --video v.mp4 \
    --gct-conf 0.5 --gct-subsample 1 --gct-fps 15
```

**Fast preview:**
```bash
eridian --gct --gct-model ckpt.pt --video v.mp4 \
    --gct-fps 3 --gct-max-frames 50 --gct-subsample 8
```

**Low memory:**
```bash
eridian --gct --gct-model ckpt.pt --video v.mp4 \
    --gct-window 16 --gct-fps 5 --gct-subsample 4
```

### Confidence threshold

The `--gct-conf` parameter is the most impactful tuning knob:

| Value | Effect |
|-------|--------|
| 0.1 - 0.5 | Keep almost everything. Noisy but complete. |
| 1.0 - 1.5 | Balanced (default range). Good for most scenes. |
| 2.0 - 3.0 | Only high-confidence points. Clean but may have gaps. |
| 5.0+ | Very sparse. Only the most reliable geometry. |

### Subsample factor

| Value | Points per frame | Use case |
|-------|-----------------|----------|
| 1 | ~268k (518x518) | Maximum density, slow to render |
| 2 | ~67k | High quality |
| 4 | ~17k | Default, good balance |
| 8 | ~4k | Fast preview |

## Viewing Point Clouds

### MeshLab
Best for quick viewing. Open the `.ply` file directly. Use the trackball to navigate.

### CloudCompare
Best for measurement and analysis. Can compute distances, fit planes, and segment regions.

### Blender
Best for rendering. Import via File > Import > PLY. Add lighting and materials for photorealistic renders.

### Open3D (Python)
```python
import open3d as o3d
pcd = o3d.io.read_point_cloud("cloud.ply")
o3d.visualization.draw_geometries([pcd])
```

## Common Issues

| Problem | Cause | Fix |
|---------|-------|-----|
| Empty point cloud | Confidence too high | Lower `--gct-conf` |
| Noisy / scattered points | Confidence too low or bad video | Raise `--gct-conf`, re-record with better lighting |
| Out of memory | Too many frames or large window | Reduce `--gct-fps`, `--gct-window`, or `--gct-max-frames` |
| Drift in classic mode | Fast motion or featureless area | Move slower, ensure textured surfaces are visible |
| Point cloud is upside down | Coordinate convention | Eridian flips Y to camera POV by default |
