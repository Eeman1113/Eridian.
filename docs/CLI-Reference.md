# CLI Reference

## Classic Mode

```bash
eridian [OPTIONS]
```

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

### Examples

```bash
eridian                              # webcam with interactive picker
eridian --camera 0                   # specific camera
eridian --video room.mp4 --save      # video file, save PLY
eridian --render --video room.mp4    # 4-panel demo video
```

## GCT Mode (LingBot-MAP)

```bash
eridian --gct --gct-model PATH --video PATH [OPTIONS]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--gct` | — | Enable GCT backend |
| `--gct-model PATH` | required | Path to GCT checkpoint (.pt) |
| `--gct-fps N` | 10 | Frame sampling rate from video |
| `--gct-max-frames N` | all | Max frames to process |
| `--gct-scale-frames N` | 8 | Scale calibration frames |
| `--gct-window N` | 64 | KV cache sliding window size |
| `--gct-conf FLOAT` | 1.5 | Point confidence threshold |
| `--gct-subsample N` | 4 | Spatial subsampling (1=full, 4=1/16th) |
| `--gct-image-size N` | 518 | Input resolution |
| `--gct-keyframe-interval N` | auto | KV cache keyframe interval |

### Examples

```bash
# Basic reconstruction
eridian --gct --gct-model ckpt.pt --video room.mp4

# High quality
eridian --gct --gct-model ckpt.pt --video room.mp4 --gct-conf 1.0 --gct-subsample 2

# Fast / low memory
eridian --gct --gct-model ckpt.pt --video room.mp4 --gct-fps 5 --gct-window 32

# Save to specific directory
eridian --gct --gct-model ckpt.pt --video room.mp4 --savedir ~/maps/
```

## Controls (Classic Mode)

| Key | Action |
|-----|--------|
| `q` | Quit |
| `Ctrl+C` | Graceful shutdown with final save |
| Mouse | Orbit / zoom / pan in 3D window |

## Output Files

### Classic
| Path | What |
|------|------|
| `splat/cloud_latest.ply` | Latest point cloud (auto-saved every 10s) |
| `splat/cloud_YYYYMMDD_HHMMSS.ply` | Timestamped backups (every 60s) |
| `splat/cloud_final_*.ply` | Final save on shutdown |
| `splat/depth_frames/*.png` | 16-bit metric depth maps |
| `output_video/eridian_demo.mp4` | 4-panel demo video |
| `logs/mapper.log` | Application log |

### GCT
| Path | What |
|------|------|
| `splat/gct_cloud.ply` | Reconstructed 3D point cloud |
| `eridian_gct_cloud.ply` | Copy in working directory (with `--save`) |
