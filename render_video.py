#!/usr/bin/env python3
"""
Render a 4-panel demo video: Camera | Depth | Features | 3D Point Cloud
Processes the test video and composites all views into a single output video.
"""

import sys
import time
import math
import numpy as np
import cv2
from pathlib import Path

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from main import DepthEstimator, PoseEstimator, PointCloud, log

# ── Config ─────────────────────────────────────────────────────────────────────
INPUT_VIDEO = BASE_DIR / "data" / "video.mp4"
OUTPUT_DIR = BASE_DIR / "output_video"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / "eridian_demo.mp4"

PANEL_W, PANEL_H = 640, 480
GRID_W, GRID_H = PANEL_W * 2, PANEL_H * 2


def render_pointcloud_view(points, colors, frame_idx, total_frames,
                           width=PANEL_W, height=PANEL_H):
    """Render the 3D point cloud with depth shading and gentle Z-axis b-roll."""
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    if len(points) < 10:
        cv2.putText(canvas, "Accumulating points...",
                    (width // 6, height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
        return canvas

    # Center the point cloud
    centroid = points.mean(axis=0)
    pts = points - centroid

    # Auto-scale to fit view
    spread = np.percentile(np.abs(pts), 95)
    if spread < 1e-6:
        spread = 1.0

    # Gentle Z-axis rotation — slow b-roll drift
    angle = (frame_idx / max(total_frames, 1)) * math.pi * 0.15
    cos_a, sin_a = math.cos(angle), math.sin(angle)

    rx = pts[:, 0] * cos_a - pts[:, 1] * sin_a
    ry = pts[:, 0] * sin_a + pts[:, 1] * cos_a
    rz = pts[:, 2]

    # Perspective projection
    focal = width * 0.8
    cam_dist = spread * 3.0
    z_shifted = rz + cam_dist

    valid = z_shifted > 0.1
    rx, ry, z_shifted = rx[valid], ry[valid], z_shifted[valid]
    cols_valid = colors[valid].astype(np.float32)

    if len(rx) == 0:
        return canvas

    px = (rx * focal / z_shifted + width / 2).astype(np.int32)
    py = (ry * focal / z_shifted + height / 2).astype(np.int32)

    # ── Depth shading: farther points are dimmer ───────────────────
    z_min, z_max = z_shifted.min(), z_shifted.max()
    z_range = z_max - z_min + 1e-6
    depth_factor = 1.0 - 0.6 * (z_shifted - z_min) / z_range  # [0.4, 1.0]
    cols_shaded = cols_valid * depth_factor[:, np.newaxis]
    cols_shaded = np.clip(cols_shaded, 0, 255).astype(np.uint8)

    # Depth sort (draw far points first, close points on top)
    depth_order = np.argsort(-z_shifted)
    px, py = px[depth_order], py[depth_order]
    cols_shaded = cols_shaded[depth_order]

    # Filter to visible region (with margin for point splats)
    mask = (px >= 1) & (px < width - 1) & (py >= 1) & (py < height - 1)
    px, py = px[mask], py[mask]
    cols_shaded = cols_shaded[mask]

    # Deterministic subsample to avoid flicker
    if len(px) > 250000:
        step = max(1, len(px) // 250000)
        px, py, cols_shaded = px[::step], py[::step], cols_shaded[::step]

    # ── Draw 3x3 point splats for visibility ───────────────────────
    bgr = cols_shaded[:, ::-1]  # RGB -> BGR
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            canvas[py + dy, px + dx] = bgr

    return canvas


def add_label(panel, label, bg_color=(0, 0, 0)):
    """Add a label bar at the top of a panel."""
    bar = np.full((32, panel.shape[1], 3), bg_color, dtype=np.uint8)
    cv2.putText(bar, label, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2)
    return np.vstack([bar, panel[:panel.shape[0] - 32]])


def main():
    log.info("=" * 60)
    log.info("Rendering 4-panel demo video (v2)")
    log.info("=" * 60)

    cap = cv2.VideoCapture(str(INPUT_VIDEO))
    if not cap.isOpened():
        log.error(f"Cannot open {INPUT_VIDEO}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    in_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    log.info(f"Input: {INPUT_VIDEO.name}, {total_frames} frames @ {in_fps:.0f}fps")

    # Use H.264 for macOS QuickTime compatibility
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(str(OUTPUT_PATH), fourcc, in_fps, (GRID_W, GRID_H))
    if not out.isOpened():
        # Fallback to mp4v if avc1 not available
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(OUTPUT_PATH), fourcc, in_fps, (GRID_W, GRID_H))

    log.info("Loading depth model...")
    depth_est = DepthEstimator()

    K_fx = PANEL_W * 0.8
    K = np.array([[K_fx, 0, PANEL_W / 2],
                  [0, K_fx, PANEL_H / 2],
                  [0, 0, 1]], dtype=np.float64)
    pose_est = PoseEstimator(K_fx, K_fx, PANEL_W / 2, PANEL_H / 2)
    cloud = PointCloud()

    # Keyframe state for render pipeline
    last_kf_pose = np.eye(4)
    frames_since_kf = 0

    frame_idx = 0
    t_start = time.time()

    cached_pts = np.zeros((0, 3), dtype=np.float32)
    cached_cols = np.zeros((0, 3), dtype=np.uint8)
    cloud_update_interval = 5

    while True:
        ok, raw_frame = cap.read()
        if not ok:
            break

        frame = cv2.resize(raw_frame, (PANEL_W, PANEL_H))
        frame_idx += 1

        # ── Depth ──────────────────────────────────────────────────
        try:
            depth = depth_est.estimate(frame)
        except Exception as e:
            log.warning(f"Frame {frame_idx}: depth failed: {e}")
            depth = np.ones((PANEL_H, PANEL_W), dtype=np.float32) * 5.0

        # ── Pose ───────────────────────────────────────────────────
        try:
            pose, kp, matches_img = pose_est.update(frame, depth=depth)
        except Exception:
            pose = np.eye(4)
            matches_img = frame.copy()

        # ── Keyframe-gated point cloud accumulation ────────────────
        frames_since_kf += 1
        t_diff = np.linalg.norm(pose[:3, 3] - last_kf_pose[:3, 3])
        R_diff = pose[:3, :3] @ last_kf_pose[:3, :3].T
        angle_diff = np.arccos(np.clip((np.trace(R_diff) - 1.0) / 2.0, -1.0, 1.0))

        if t_diff > 0.08 or angle_diff > np.radians(5) or frames_since_kf > 10:
            try:
                cloud.add_frame(depth, frame, pose, K)
            except Exception:
                pass
            last_kf_pose = pose.copy()
            frames_since_kf = 0

        # ── Build 4 panels ─────────────────────────────────────────
        elapsed = time.time() - t_start
        fps = frame_idx / elapsed if elapsed > 0 else 0

        # Panel 1: Camera feed
        p1 = frame.copy()
        cv2.putText(p1, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(p1, f"Frame: {frame_idx}/{total_frames}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        p1 = add_label(p1, "CAMERA")

        # Panel 2: Depth map
        d_norm = ((depth - depth.min()) / (depth.max() - depth.min() + 1e-6) * 255).astype(np.uint8)
        p2 = cv2.applyColorMap(d_norm, cv2.COLORMAP_INFERNO)
        p2 = add_label(p2, "DEPTH (metric)")

        # Panel 3: Features
        p3 = matches_img.copy()
        cv2.putText(p3, f"Tracking: {'OK' if pose_est.tracking else 'LOST'}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0) if pose_est.tracking else (0, 0, 255), 2)
        p3 = add_label(p3, "OPTICAL FLOW + PnP")

        # Panel 4: 3D point cloud — cached for stability
        if frame_idx % cloud_update_interval == 1 or frame_idx == 1:
            cached_pts, cached_cols = cloud.get_data()
        p4 = render_pointcloud_view(cached_pts, cached_cols, frame_idx, total_frames)
        cv2.putText(p4, f"Points: {len(cached_pts):,}", (10, PANEL_H - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        p4 = add_label(p4, "3D POINT CLOUD")

        # ── Compose ────────────────────────────────────────────────
        top = np.hstack([p1, p2])
        bot = np.hstack([p3, p4])
        out.write(np.vstack([top, bot]))

        if frame_idx % 10 == 0 or frame_idx == total_frames:
            pct = frame_idx / total_frames * 100
            log.info(f"  [{pct:5.1f}%] Frame {frame_idx}/{total_frames} | "
                     f"Points: {cloud.count():,} | {fps:.1f} fps")

    cap.release()
    out.release()

    duration = time.time() - t_start
    file_size = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    log.info("=" * 60)
    log.info(f"Done! {frame_idx} frames in {duration:.1f}s")
    log.info(f"Output: {OUTPUT_PATH} ({file_size:.1f} MB)")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
