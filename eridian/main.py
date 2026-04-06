#!/usr/bin/env python3
"""
Eridian — Real-time monocular 3D point cloud reconstruction.

Uses webcam + metric depth estimation + PnP visual odometry to build
a colored 3D point cloud of the environment in real time.
"""

import argparse
import dataclasses
import sys
import time
import signal
import logging
import datetime
from pathlib import Path
from threading import Lock

import cv2
import numpy as np
import torch

# ── Paths ──────────────────────────────────────────────────────────────────────
# Output dirs use CWD (works for both clone-and-run and pip install)
BASE_DIR = Path.cwd()
SPLAT_DIR = BASE_DIR / "splat"
DEPTH_DIR = SPLAT_DIR / "depth_frames"
LOG_DIR = BASE_DIR / "logs"

# Test video: check CWD first, then repo root relative to package
_cwd_video = BASE_DIR / "data" / "video.mp4"
_pkg_video = Path(__file__).parent.parent / "data" / "video.mp4"
TEST_VIDEO = _cwd_video if _cwd_video.exists() else _pkg_video

# ── Logging (deferred to avoid side effects on import) ─────────────────────────
log = logging.getLogger("mapper")
_log_initialized = False


def _init_runtime():
    """Create output dirs and configure logging. Called once before running."""
    global _log_initialized
    if _log_initialized:
        return
    for d in [SPLAT_DIR, DEPTH_DIR, LOG_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(LOG_DIR / "mapper.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    _log_initialized = True


# ============================================================================
# 1. CAMERA DISCOVERY + CAPTURE
# ============================================================================
def probe_cameras(max_index=9):
    """Scan for available cameras. Returns list of dicts with index and resolution."""
    cameras = []
    for idx in range(max_index + 1):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cameras.append({"index": idx, "width": w, "height": h})
            cap.release()
        else:
            cap.release()
    return cameras


def pick_camera(cameras):
    """Interactive camera picker. Returns selected camera index or None."""
    if not cameras:
        print("No cameras found.")
        return None

    if len(cameras) == 1:
        c = cameras[0]
        print(f"Found 1 camera: index {c['index']} ({c['width']}x{c['height']})")
        return c["index"]

    print(f"\nFound {len(cameras)} cameras:\n")
    for i, c in enumerate(cameras):
        print(f"  [{i}] Camera {c['index']} ({c['width']}x{c['height']})")
    print()

    try:
        choice = input("Select camera number: ").strip()
        idx = int(choice)
        if 0 <= idx < len(cameras):
            return cameras[idx]["index"]
        print(f"Invalid selection: {idx}")
        return cameras[0]["index"]
    except (ValueError, KeyboardInterrupt, EOFError):
        return None


class CameraCapture:
    """OpenCV camera with auto-fallback to video file."""

    def __init__(self, width=640, height=480, fps=30, video_path=None, loop=True,
                 camera_index=None):
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None
        self.is_video = False
        self.video_path = video_path
        self.camera_index = camera_index
        self._loop = loop
        self._open()

    def _open(self):
        if self.video_path:
            return self._open_video(self.video_path)

        indices = [self.camera_index] if self.camera_index is not None else [0, 1, 2]
        for idx in indices:
            log.info(f"Trying camera index {idx}...")
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                ok, _ = cap.read()
                if ok:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                    cap.set(cv2.CAP_PROP_FPS, self.fps)
                    self.cap = cap
                    self.is_video = False
                    log.info(f"Camera opened on index {idx}")
                    return
            cap.release()

        log.warning("No camera found, looking for test video...")
        if TEST_VIDEO.exists():
            self._open_video(str(TEST_VIDEO))
        else:
            log.error("No camera and no test video at ./data/video.mp4")
            self.cap = None

    def _open_video(self, path):
        cap = cv2.VideoCapture(str(path))
        if cap.isOpened():
            self.cap = cap
            self.is_video = True
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            vfps = cap.get(cv2.CAP_PROP_FPS)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            log.info(f"TEST MODE: Using video {path} ({w}x{h}, {total} frames, {vfps:.0f}fps)")
        else:
            log.error(f"Cannot open video: {path}")
            self.cap = None

    def read(self):
        if self.cap is None or not self.cap.isOpened():
            if not self.is_video:
                log.warning("Camera disconnected, attempting reconnect...")
                self._open()
            if self.cap is None:
                return False, None

        ok, frame = self.cap.read()

        if not ok and self.is_video:
            if self._loop:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self.cap.read()
                log.info("Test video looped")
            else:
                return False, None

        if not ok and not self.is_video:
            log.warning("Frame read failed, reconnecting...")
            self.cap.release()
            self.cap = None
            self._open()
            if self.cap is not None:
                ok, frame = self.cap.read()

        if ok and frame is not None:
            h, w = frame.shape[:2]
            if w != self.width or h != self.height:
                frame = cv2.resize(frame, (self.width, self.height))

        return ok, frame

    def release(self):
        if self.cap is not None:
            self.cap.release()


# ============================================================================
# 2. MONOCULAR DEPTH ESTIMATION (with bilateral + temporal smoothing)
# ============================================================================
class DepthEstimator:
    """Depth Anything V2 Metric with bilateral filtering and temporal EMA."""

    def __init__(self):
        self.pipe = None
        self.model_name = None
        self.is_metric = False
        self.prev_depth = None
        self._load_model()

    def _load_model(self):
        from transformers import pipeline

        device = "mps" if torch.backends.mps.is_available() else "cpu"

        candidates = [
            ("depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf", True),
            ("depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf", True),
            ("depth-anything/Depth-Anything-V2-Small-hf", False),
            ("Intel/dpt-swinv2-tiny-256", False),
            ("Intel/dpt-hybrid-midas", False),
        ]
        for name, metric in candidates:
            try:
                log.info(f"Loading depth model: {name}")
                self.pipe = pipeline("depth-estimation", model=name, device=device)
                self.model_name = name
                self.is_metric = metric
                log.info(f"Depth model loaded: {name} ({'METRIC' if metric else 'relative'})")
                return
            except Exception as e:
                log.warning(f"Failed to load {name}: {e}")

        raise RuntimeError("Could not load any depth estimation model")

    def estimate(self, frame_bgr):
        """Returns depth map as float32 HxW array in meters, filtered and smoothed."""
        from PIL import Image as PILImage

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = PILImage.fromarray(rgb)

        result = self.pipe(pil_img)

        if self.is_metric and "predicted_depth" in result:
            depth = result["predicted_depth"]
            if hasattr(depth, 'cpu'):
                depth = depth.cpu().numpy()
            depth = np.array(depth, dtype=np.float32)
        else:
            depth_raw = result["depth"]
            if isinstance(depth_raw, PILImage.Image):
                depth = np.array(depth_raw, dtype=np.float32)
            else:
                depth = np.array(depth_raw, dtype=np.float32)
            d_min, d_max = depth.min(), depth.max()
            if d_max - d_min > 1e-6:
                depth = (depth - d_min) / (d_max - d_min)
            depth = 0.5 + depth * 9.5

        h, w = frame_bgr.shape[:2]
        if depth.shape != (h, w):
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)

        # ── Bilateral filter: smooth noise, preserve edges ─────────
        depth = cv2.bilateralFilter(depth, d=7, sigmaColor=0.3, sigmaSpace=7)

        # ── Temporal EMA: reduce per-frame jitter ──────────────────
        alpha = 0.65
        if self.prev_depth is not None and self.prev_depth.shape == depth.shape:
            depth = alpha * depth + (1.0 - alpha) * self.prev_depth
        self.prev_depth = depth.copy()

        return depth


# ============================================================================
# 3. POSE ESTIMATION (GFTT + Lucas-Kanade + PnP)
# ============================================================================
class PoseEstimator:
    """Visual odometry using optical flow tracking and PnP from depth."""

    def __init__(self, fx, fy, cx, cy):
        self.K = np.array([[fx, 0, cx],
                           [0, fy, cy],
                           [0,  0,  1]], dtype=np.float64)
        self.fx, self.fy = fx, fy
        self.cx, self.cy = cx, cy

        # GFTT corner detection params
        self.gftt_params = dict(
            maxCorners=500, qualityLevel=0.01,
            minDistance=15, blockSize=7,
        )

        # Lucas-Kanade optical flow params
        self.lk_params = dict(
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )

        self.prev_gray = None
        self.prev_pts = None
        self.prev_depth = None
        self.pose = np.eye(4, dtype=np.float64)
        self.tracking = True
        self.frame_count = 0

    def update(self, frame_bgr, depth=None):
        """Update pose via LK optical flow + PnP. Returns (pose, kp_list, viz_img)."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        self.frame_count += 1

        viz = frame_bgr.copy()

        # ── Detect features if needed ──────────────────────────────
        if self.prev_pts is None or len(self.prev_pts) < 80 or self.frame_count % 30 == 0:
            pts = cv2.goodFeaturesToTrack(gray, **self.gftt_params)
            if pts is None:
                self.prev_gray = gray
                self.prev_pts = None
                self.prev_depth = depth
                return self.pose.copy(), [], viz

            self.prev_gray = gray
            self.prev_pts = pts
            self.prev_depth = depth

            # Draw detected features
            for p in pts:
                x, y = int(p[0, 0]), int(p[0, 1])
                cv2.circle(viz, (x, y), 3, (0, 255, 0), -1)

            return self.pose.copy(), [], viz

        # ── Track features with LK optical flow ───────────────────
        pts_next, status, err = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.prev_pts, None, **self.lk_params
        )

        if pts_next is None:
            self.tracking = False
            self.prev_gray = gray
            self.prev_pts = None
            self.prev_depth = depth
            return self.pose.copy(), [], viz

        # ── Forward-backward consistency check ─────────────────────
        pts_back, status_back, _ = cv2.calcOpticalFlowPyrLK(
            gray, self.prev_gray, pts_next, None, **self.lk_params
        )

        status = status.ravel().astype(bool)
        if pts_back is not None:
            status_back = status_back.ravel().astype(bool)
            fb_err = np.linalg.norm(
                self.prev_pts.reshape(-1, 2) - pts_back.reshape(-1, 2), axis=1
            )
            # Keep only points that survived both directions with < 1px round-trip error
            good_mask = status & status_back & (fb_err < 1.0)
        else:
            good_mask = status

        prev_good = self.prev_pts[good_mask].reshape(-1, 2)
        curr_good = pts_next[good_mask].reshape(-1, 2)

        # ── Draw optical flow trails ───────────────────────────────
        for i in range(len(curr_good)):
            x0, y0 = int(prev_good[i, 0]), int(prev_good[i, 1])
            x1, y1 = int(curr_good[i, 0]), int(curr_good[i, 1])
            cv2.line(viz, (x0, y0), (x1, y1), (0, 255, 255), 1)
            cv2.circle(viz, (x1, y1), 2, (0, 255, 0), -1)

        if len(prev_good) < 10:
            log.warning(f"Tracking: only {len(prev_good)} points after FB check")
            self.tracking = False
            self.prev_gray = gray
            self.prev_pts = cv2.goodFeaturesToTrack(gray, **self.gftt_params)
            self.prev_depth = depth
            return self.pose.copy(), [], viz

        # ── PnP pose from depth ────────────────────────────────────
        if self.prev_depth is not None:
            # Lift previous 2D points to 3D using depth
            px = np.clip(prev_good[:, 0].astype(int), 0, self.prev_depth.shape[1] - 1)
            py = np.clip(prev_good[:, 1].astype(int), 0, self.prev_depth.shape[0] - 1)
            z = self.prev_depth[py, px]

            # Filter valid depth
            valid = (z > 0.2) & (z < 12.0)
            if valid.sum() < 8:
                self.tracking = False
                self.prev_gray = gray
                self.prev_pts = cv2.goodFeaturesToTrack(gray, **self.gftt_params)
                self.prev_depth = depth
                return self.pose.copy(), [], viz

            z_valid = z[valid]
            prev_2d = prev_good[valid]
            curr_2d = curr_good[valid]

            # Back-project to 3D camera coords (previous frame)
            x3d = (prev_2d[:, 0] - self.cx) * z_valid / self.fx
            y3d = (prev_2d[:, 1] - self.cy) * z_valid / self.fy
            pts_3d = np.stack([x3d, y3d, z_valid], axis=1).astype(np.float64)
            pts_2d = curr_2d.astype(np.float64)

            # Solve PnP — gives pose of current frame relative to previous 3D points
            ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                pts_3d, pts_2d, self.K, None,
                iterationsCount=200, reprojectionError=3.0,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )

            if ok and inliers is not None and len(inliers) >= 8:
                R, _ = cv2.Rodrigues(rvec)
                T_rel = np.eye(4, dtype=np.float64)
                T_rel[:3, :3] = R
                T_rel[:3, 3] = tvec.ravel()

                self.pose = self.pose @ np.linalg.inv(T_rel)
                self.tracking = True

                # Highlight inliers
                for idx in inliers.ravel():
                    if idx < len(curr_2d):
                        x, y = int(curr_2d[idx, 0]), int(curr_2d[idx, 1])
                        cv2.circle(viz, (x, y), 4, (255, 0, 255), 1)
            else:
                log.warning(f"PnP failed or too few inliers ({0 if inliers is None else len(inliers)})")
                self.tracking = False

        # ── Update state ───────────────────────────────────────────
        # Re-detect if too few tracked points
        if len(curr_good) < 100:
            self.prev_pts = cv2.goodFeaturesToTrack(gray, **self.gftt_params)
        else:
            self.prev_pts = curr_good.reshape(-1, 1, 2).astype(np.float32)
        self.prev_gray = gray
        self.prev_depth = depth

        return self.pose.copy(), [], viz


# ============================================================================
# 4. POINT CLOUD (with edge/normal filtering + voxel averaging)
# ============================================================================
class PointCloud:
    """Global point cloud with depth-edge filtering, normal rejection, and voxel averaging."""

    def __init__(self, max_points=2_000_000, voxel_size=0.03):
        self.points = np.zeros((0, 3), dtype=np.float32)
        self.colors = np.zeros((0, 3), dtype=np.uint8)
        self.max_points = max_points
        self.voxel_size = voxel_size
        self.lock = Lock()
        self.frame_count = 0

    def add_frame(self, depth, frame_bgr, pose, K, subsample=4):
        """Back-project depth to 3D with quality filtering, transform, append."""
        h, w = depth.shape
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        # ── Depth gradients (for edge + normal filtering) ──────────
        grad_x = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

        # ── Subsample grid ─────────────────────────────────────────
        ys = np.arange(0, h, subsample)
        xs = np.arange(0, w, subsample)
        xs, ys = np.meshgrid(xs, ys)
        xs, ys = xs.ravel(), ys.ravel()

        z = depth[ys, xs]
        gm = grad_mag[ys, xs]

        # ── Filter 1: valid depth range ────────────────────────────
        valid = (z > 0.2) & (z < 12.0)

        # ── Filter 2: depth edge rejection ─────────────────────────
        # Reject points where depth gradient > 15% of depth (flying pixels)
        edge_thresh = 0.15 * z
        valid &= (gm < edge_thresh)

        # ── Filter 3: grazing angle rejection via surface normals ──
        gx = grad_x[ys, xs]
        gy = grad_y[ys, xs]
        # Surface normal in camera frame from depth gradients
        nx = -gx / fx
        ny = -gy / fy
        nz = np.ones_like(nx)
        n_norm = np.sqrt(nx ** 2 + ny ** 2 + nz ** 2) + 1e-8
        nx, ny, nz = nx / n_norm, ny / n_norm, nz / n_norm

        # Camera ray direction per pixel
        ray_x = (xs.astype(np.float32) - cx) / fx
        ray_y = (ys.astype(np.float32) - cy) / fy
        ray_z = np.ones_like(ray_x)
        ray_norm = np.sqrt(ray_x ** 2 + ray_y ** 2 + ray_z ** 2)
        ray_x, ray_y, ray_z = ray_x / ray_norm, ray_y / ray_norm, ray_z / ray_norm

        # Dot product: cos(angle between normal and view ray)
        cos_angle = np.abs(nx * ray_x + ny * ray_y + nz * ray_z)
        valid &= (cos_angle > 0.25)  # reject angles > ~75 degrees

        # ── Apply filter ───────────────────────────────────────────
        xs, ys, z = xs[valid], ys[valid], z[valid]

        if len(z) < 10:
            return

        # ── Back-project to 3D camera coords ──────────────────────
        x3d = (xs.astype(np.float32) - cx) * z / fx
        y3d = (ys.astype(np.float32) - cy) * z / fy
        pts_cam = np.stack([x3d, y3d, z], axis=1)

        # ── Transform to world coordinates ─────────────────────────
        R = pose[:3, :3].astype(np.float32)
        t = pose[:3, 3].astype(np.float32)
        pts_world = (pts_cam @ R.T) + t

        # ── Colors (BGR -> RGB) ────────────────────────────────────
        colors = frame_bgr[ys, xs][:, ::-1].copy()

        with self.lock:
            self.points = np.vstack([self.points, pts_world])
            self.colors = np.vstack([self.colors, colors])
            self.frame_count += 1

            if self.frame_count % 20 == 0:
                self._voxel_downsample()

    def _voxel_downsample(self):
        """Voxel averaging: merge points in the same voxel by averaging positions and colors."""
        if len(self.points) < 1000:
            return

        n_before = len(self.points)

        voxel_idx = np.floor(self.points / self.voxel_size).astype(np.int64)
        voxel_idx -= voxel_idx.min(axis=0)
        dims = voxel_idx.max(axis=0) + 1
        flat = (voxel_idx[:, 0] * dims[1] * dims[2]
                + voxel_idx[:, 1] * dims[2]
                + voxel_idx[:, 2])

        unique_voxels, inverse = np.unique(flat, return_inverse=True)
        n_voxels = len(unique_voxels)

        # Average positions and colors per voxel using bincount
        counts = np.bincount(inverse, minlength=n_voxels).astype(np.float32)
        avg_pts = np.zeros((n_voxels, 3), dtype=np.float32)
        avg_cols = np.zeros((n_voxels, 3), dtype=np.float32)
        for dim in range(3):
            avg_pts[:, dim] = np.bincount(
                inverse, weights=self.points[:, dim], minlength=n_voxels
            ) / counts
            avg_cols[:, dim] = np.bincount(
                inverse, weights=self.colors[:, dim].astype(np.float32), minlength=n_voxels
            ) / counts

        self.points = avg_pts
        self.colors = np.clip(avg_cols, 0, 255).astype(np.uint8)

        # If over budget, keep highest-density voxels
        if len(self.points) > self.max_points:
            keep = np.argsort(-counts)[:self.max_points]
            self.points = self.points[keep]
            self.colors = self.colors[keep]

        log.info(f"Voxel average: {n_before} -> {len(self.points)} points")

    def get_data(self):
        with self.lock:
            return self.points.copy(), self.colors.copy()

    def count(self):
        return len(self.points)


# ============================================================================
# 5. PLY EXPORT
# ============================================================================
def save_ply(filepath, points, colors):
    """Save point cloud as binary PLY."""
    n = len(points)
    if n == 0:
        log.warning("No points to save")
        return

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )

    dtype = np.dtype([
        ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
        ('r', 'u1'), ('g', 'u1'), ('b', 'u1'),
    ])
    data = np.empty(n, dtype=dtype)
    data['x'] = points[:, 0]
    data['y'] = points[:, 1]
    data['z'] = points[:, 2]
    data['r'] = colors[:, 0]
    data['g'] = colors[:, 1]
    data['b'] = colors[:, 2]

    with open(str(filepath), 'wb') as f:
        f.write(header.encode('ascii'))
        f.write(data.tobytes())

    log.info(f"Saved {n} points to {filepath}")


def load_ply(filepath):
    """Load a binary PLY file. Returns (points_Nx3 float32, colors_Nx3 uint8)."""
    with open(str(filepath), 'rb') as f:
        header = b""
        n_vertices = 0
        while True:
            line = f.readline()
            header += line
            text = line.decode('ascii', errors='ignore').strip()
            if text.startswith("element vertex"):
                n_vertices = int(text.split()[-1])
            if text == "end_header":
                break

        dtype = np.dtype([
            ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
            ('r', 'u1'), ('g', 'u1'), ('b', 'u1'),
        ])
        data = np.frombuffer(f.read(n_vertices * dtype.itemsize), dtype=dtype)

    points = np.stack([data['x'], data['y'], data['z']], axis=1)
    colors = np.stack([data['r'], data['g'], data['b']], axis=1)
    return points, colors


# ============================================================================
# 6. 3D VISUALIZER (PyVista)
# ============================================================================
class Visualizer3D:
    """Non-blocking 3D point cloud viewer."""

    def __init__(self):
        self.plotter = None
        self.actor = None
        self._initialized = False

    def init(self):
        try:
            import pyvista as pv
            pv.global_theme.background = 'black'
            self.plotter = pv.Plotter(title="3D World Map", window_size=(960, 720))
            self.plotter.show(interactive_update=True, auto_close=False)
            self._initialized = True
            log.info("3D visualizer initialized")
        except Exception as e:
            log.warning(f"Could not initialize 3D visualizer: {e}")
            self._initialized = False

    def update(self, points, colors):
        if not self._initialized or len(points) < 10:
            return
        try:
            import pyvista as pv
            cloud = pv.PolyData(points.astype(np.float32))
            cloud.point_data['RGB'] = colors
            if self.actor is not None:
                self.plotter.remove_actor(self.actor)
            self.actor = self.plotter.add_points(
                cloud, scalars='RGB', rgb=True,
                point_size=2, render_points_as_spheres=False,
            )
            self.plotter.update()
        except Exception as e:
            log.debug(f"Viz update error: {e}")

    def close(self):
        if self._initialized and self.plotter is not None:
            try:
                self.plotter.close()
            except Exception:
                pass


# ============================================================================
# 7. MAIN APPLICATION
# ============================================================================
class WorldMapper:
    """Main loop with keyframe-based accumulation."""

    def __init__(self, video_path=None, headless=False, camera_index=None, save=False):
        self.running = True
        self.camera = None
        self.depth_estimator = None
        self.pose_estimator = None
        self.cloud = PointCloud()
        self.viz3d = Visualizer3D()
        self.video_path = video_path
        self.headless = headless
        self.camera_index = camera_index
        self.save_to_cwd = save

        # Timing
        self.last_save = time.time()
        self.last_backup = time.time()
        self.last_status = time.time()
        self.frame_idx = 0
        self.fps_history = []

        # Keyframe state
        self.last_kf_pose = np.eye(4, dtype=np.float64)
        self.frames_since_kf = 0
        self.keyframe_count = 0

        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, sig, frame):
        log.info("SIGINT received, shutting down gracefully...")
        self.running = False

    def _is_keyframe(self, pose):
        """Decide if this frame should contribute points to the cloud."""
        self.frames_since_kf += 1

        t_diff = np.linalg.norm(pose[:3, 3] - self.last_kf_pose[:3, 3])
        R_diff = pose[:3, :3] @ self.last_kf_pose[:3, :3].T
        angle = np.arccos(np.clip((np.trace(R_diff) - 1.0) / 2.0, -1.0, 1.0))

        # Keyframe if moved enough, rotated enough, or been too long
        if t_diff > 0.08 or angle > np.radians(5) or self.frames_since_kf > 15:
            self.last_kf_pose = pose.copy()
            self.frames_since_kf = 0
            self.keyframe_count += 1
            return True
        return False

    def _save_state(self, tag=""):
        pts, cols = self.cloud.get_data()
        if len(pts) == 0:
            return
        save_ply(SPLAT_DIR / "cloud_latest.ply", pts, cols)
        if tag:
            save_ply(SPLAT_DIR / f"cloud_{tag}.ply", pts, cols)
        if self.save_to_cwd:
            save_ply(Path.cwd() / "eridian_cloud.ply", pts, cols)

    def run(self):
        _init_runtime()
        log.info("=" * 60)
        log.info("Eridian v2 — 3D World Mapper")
        log.info("=" * 60)

        # ── Init camera ────────────────────────────────────────────
        log.info("Initializing camera...")
        loop = not self.headless
        self.camera = CameraCapture(video_path=self.video_path, loop=loop,
                                    camera_index=self.camera_index)
        if self.camera.cap is None:
            log.error("No camera available. Exiting.")
            return

        ok, test_frame = self.camera.read()
        if not ok:
            log.error("Cannot read from camera. Exiting.")
            return

        h, w = test_frame.shape[:2]
        log.info(f"Camera resolution: {w}x{h}")

        # ── Camera intrinsics ──────────────────────────────────────
        fx = fy = w * 0.8
        cx, cy = w / 2.0, h / 2.0
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        log.info(f"Camera intrinsics: fx={fx:.0f} fy={fy:.0f} cx={cx:.0f} cy={cy:.0f}")

        # ── Init modules ───────────────────────────────────────────
        log.info("Loading depth estimation model...")
        try:
            self.depth_estimator = DepthEstimator()
        except Exception as e:
            log.error(f"Failed to load depth model: {e}")
            return

        self.pose_estimator = PoseEstimator(fx, fy, cx, cy)

        if not self.headless:
            self.viz3d.init()
            cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Depth", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Features", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Camera", 480, 360)
            cv2.resizeWindow("Depth", 480, 360)
            cv2.resizeWindow("Features", 480, 360)
            cv2.moveWindow("Camera", 0, 0)
            cv2.moveWindow("Depth", 490, 0)
            cv2.moveWindow("Features", 980, 0)

        mode = "HEADLESS" if self.headless else "GUI"
        log.info(f"Starting main loop ({mode})")
        if not self.headless:
            log.info("Press 'q' in any OpenCV window or Ctrl+C to quit")

        # ── Main loop ──────────────────────────────────────────────
        try:
            while self.running:
                t_start = time.time()

                ok, frame = self.camera.read()
                if not ok:
                    if self.headless and self.camera.is_video:
                        log.info("Video finished, saving final output...")
                        break
                    log.warning("No frame, waiting for camera...")
                    self._save_state("disconnect")
                    time.sleep(1.0)
                    continue

                self.frame_idx += 1

                # ── Depth ──────────────────────────────────────────
                depth = None
                try:
                    depth = self.depth_estimator.estimate(frame)
                except Exception as e:
                    log.warning(f"Depth estimation failed: {e}")

                # ── Pose ───────────────────────────────────────────
                pose = None
                matches_img = frame.copy()
                try:
                    pose, kp, matches_img = self.pose_estimator.update(frame, depth=depth)
                except Exception as e:
                    log.warning(f"Pose estimation failed: {e}")

                # ── Keyframe-gated point accumulation ──────────────
                if depth is not None and pose is not None:
                    if self._is_keyframe(pose):
                        try:
                            self.cloud.add_frame(depth, frame, pose, K)
                        except Exception as e:
                            log.warning(f"Point cloud accumulation failed: {e}")

                # ── FPS ────────────────────────────────────────────
                dt = time.time() - t_start
                fps = 1.0 / dt if dt > 0 else 0
                self.fps_history.append(fps)
                if len(self.fps_history) > 30:
                    self.fps_history = self.fps_history[-30:]
                avg_fps = sum(self.fps_history) / len(self.fps_history)

                # ── Display ────────────────────────────────────────
                if not self.headless:
                    disp = frame.copy()
                    cv2.putText(disp, f"FPS: {avg_fps:.1f}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.putText(disp, f"Points: {self.cloud.count():,}", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(disp, f"KF: {self.keyframe_count}", (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
                    cv2.imshow("Camera", disp)

                    if depth is not None:
                        d_norm = ((depth - depth.min()) / (depth.max() - depth.min() + 1e-6) * 255).astype(np.uint8)
                        cv2.imshow("Depth", cv2.applyColorMap(d_norm, cv2.COLORMAP_INFERNO))

                    cv2.imshow("Features", matches_img)

                    if self.frame_idx % 5 == 0:
                        pts, cols = self.cloud.get_data()
                        self.viz3d.update(pts, cols)

                # ── Save depth frames ──────────────────────────────
                if depth is not None and self.frame_idx % 5 == 0:
                    try:
                        depth_16 = (depth * 1000).astype(np.uint16)
                        cv2.imwrite(str(DEPTH_DIR / f"depth_{self.frame_idx:06d}.png"), depth_16)
                    except Exception:
                        pass

                # ── Periodic saves ─────────────────────────────────
                now = time.time()
                if now - self.last_save >= 10:
                    self._save_state()
                    self.last_save = now
                if now - self.last_backup >= 60:
                    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    self._save_state(ts)
                    self.last_backup = now

                # ── Status ─────────────────────────────────────────
                if now - self.last_status >= 5:
                    tracking = "OK" if self.pose_estimator.tracking else "LOST"
                    log.info(
                        f"STATUS | FPS: {avg_fps:.1f} | Points: {self.cloud.count():,} | "
                        f"Tracking: {tracking} | Keyframes: {self.keyframe_count} | "
                        f"Last save: {now - self.last_save:.0f}s ago"
                    )
                    self.last_status = now

                # ── Quit ───────────────────────────────────────────
                if not self.headless:
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        log.info("'q' pressed, shutting down...")
                        self.running = False

        except Exception as e:
            log.error(f"Main loop error: {e}", exc_info=True)
        finally:
            self._shutdown()

    def _shutdown(self):
        log.info("Shutting down...")
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self._save_state(f"final_{ts}")
        if self.camera:
            self.camera.release()
        cv2.destroyAllWindows()
        self.viz3d.close()
        log.info(f"Final point count: {self.cloud.count():,}")
        log.info(f"Total keyframes: {self.keyframe_count}")
        log.info("Shutdown complete")


# ============================================================================
# 8. HIGH-LEVEL PYTHON API
# ============================================================================
@dataclasses.dataclass
class FrameResult:
    """Result from processing a single frame."""
    frame: np.ndarray
    depth: np.ndarray
    pose: np.ndarray
    points: np.ndarray
    colors: np.ndarray
    is_keyframe: bool
    frame_index: int


class Eridian:
    """High-level API for monocular 3D reconstruction.

    Usage::

        e = Eridian()

        # Single-frame depth
        depth = e.estimate_depth(frame)

        # Process a whole video
        points, colors = e.process_video("video.mp4")
        e.save("output.ply")

        # Stream from camera
        for result in e.stream(camera=0):
            print(f"Frame {result.frame_index}: {len(result.points)} points")
    """

    def __init__(self):
        self._depth_est = None
        self._pose_est = None
        self._cloud = PointCloud()
        self._K = None
        self._last_kf_pose = np.eye(4, dtype=np.float64)
        self._frames_since_kf = 0
        self._keyframe_count = 0
        self._callbacks = {"on_frame": [], "on_keyframe": [], "on_depth": []}

    def _ensure_depth(self):
        if self._depth_est is None:
            self._depth_est = DepthEstimator()

    def _ensure_pose(self, width, height):
        if self._pose_est is None:
            fx = fy = width * 0.8
            cx, cy = width / 2.0, height / 2.0
            self._K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
            self._pose_est = PoseEstimator(fx, fy, cx, cy)

    def _is_keyframe(self, pose):
        self._frames_since_kf += 1
        t_diff = np.linalg.norm(pose[:3, 3] - self._last_kf_pose[:3, 3])
        R_diff = pose[:3, :3] @ self._last_kf_pose[:3, :3].T
        angle = np.arccos(np.clip((np.trace(R_diff) - 1.0) / 2.0, -1.0, 1.0))
        if t_diff > 0.08 or angle > np.radians(5) or self._frames_since_kf > 15:
            self._last_kf_pose = pose.copy()
            self._frames_since_kf = 0
            self._keyframe_count += 1
            return True
        return False

    def _fire(self, event, *args):
        for cb in self._callbacks.get(f"on_{event}", []):
            cb(*args)

    # ── Callbacks ────────────────────────────────────────────────────

    def on(self, event, callback):
        """Register a callback. Events: 'frame', 'keyframe', 'depth'.

        Callback signatures:
          frame:    fn(frame_index, frame_bgr, depth, pose)
          keyframe: fn(frame_index, points, colors)
          depth:    fn(frame_index, depth_map)

        Returns self for chaining.
        """
        key = f"on_{event}"
        if key not in self._callbacks:
            raise ValueError(f"Unknown event '{event}'. Use: frame, keyframe, depth")
        self._callbacks[key].append(callback)
        return self

    # ── Single-frame operations ──────────────────────────────────────

    def estimate_depth(self, image):
        """Estimate depth from a BGR image (numpy array) or file path.

        Returns float32 HxW depth map in meters.

        Args:
            image: numpy array (BGR) or string/Path to an image file.
        """
        self._ensure_depth()
        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image))
            if image is None:
                raise FileNotFoundError(f"Cannot read image: {image}")
        return self._depth_est.estimate(image)

    def frame_to_points(self, image, depth=None):
        """Convert a single frame to 3D points in camera coordinates.

        Returns (points_Nx3 float32, colors_Nx3 uint8).

        Args:
            image: BGR numpy array.
            depth: optional precomputed depth map. Estimated if None.
        """
        if depth is None:
            depth = self.estimate_depth(image)
        h, w = depth.shape
        self._ensure_pose(w, h)
        fx, fy = self._K[0, 0], self._K[1, 1]
        cx, cy = self._K[0, 2], self._K[1, 2]

        ys, xs = np.mgrid[0:h:4, 0:w:4]
        ys, xs = ys.ravel(), xs.ravel()
        z = depth[ys, xs]
        valid = (z > 0.2) & (z < 12.0)
        xs, ys, z = xs[valid], ys[valid], z[valid]

        x3d = (xs.astype(np.float32) - cx) * z / fx
        y3d = (ys.astype(np.float32) - cy) * z / fy
        pts = np.stack([x3d, y3d, z], axis=1)
        colors = image[ys, xs][:, ::-1].copy()
        return pts, colors

    # ── Batch processing ─────────────────────────────────────────────

    def process_video(self, video_path, max_frames=None):
        """Process a video file end-to-end. Returns (points, colors).

        Args:
            video_path: path to video file.
            max_frames: optional limit on frames to process.
        """
        self._ensure_depth()
        cap = CameraCapture(video_path=str(video_path), loop=False)
        if cap.cap is None:
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        h, w = 480, 640
        self._ensure_pose(w, h)
        frame_idx = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            if max_frames and frame_idx > max_frames:
                break

            depth = self._depth_est.estimate(frame)
            self._fire("depth", frame_idx, depth)

            pose, _, _ = self._pose_est.update(frame, depth=depth)
            self._fire("frame", frame_idx, frame, depth, pose)

            if self._is_keyframe(pose):
                self._cloud.add_frame(depth, frame, pose, self._K)
                pts, cols = self._cloud.get_data()
                self._fire("keyframe", frame_idx, pts, cols)

        cap.release()
        return self._cloud.get_data()

    # ── Camera streaming ─────────────────────────────────────────────

    def stream(self, camera=0, max_frames=None):
        """Generator that yields FrameResult for each frame from a camera.

        Args:
            camera: camera index (int) or video file path (str).
            max_frames: optional limit.

        Yields:
            FrameResult with frame, depth, pose, accumulated points/colors.
        """
        self._ensure_depth()

        if isinstance(camera, str):
            cap = CameraCapture(video_path=camera, loop=False)
        else:
            cap = CameraCapture(camera_index=camera)
        if cap.cap is None:
            raise RuntimeError(f"Cannot open camera/video: {camera}")

        h, w = 480, 640
        self._ensure_pose(w, h)
        frame_idx = 0

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_idx += 1
                if max_frames and frame_idx > max_frames:
                    break

                depth = self._depth_est.estimate(frame)
                self._fire("depth", frame_idx, depth)

                pose, _, _ = self._pose_est.update(frame, depth=depth)
                is_kf = self._is_keyframe(pose)
                if is_kf:
                    self._cloud.add_frame(depth, frame, pose, self._K)

                self._fire("frame", frame_idx, frame, depth, pose)
                if is_kf:
                    pts, cols = self._cloud.get_data()
                    self._fire("keyframe", frame_idx, pts, cols)

                pts, cols = self._cloud.get_data()
                yield FrameResult(
                    frame=frame, depth=depth, pose=pose,
                    points=pts, colors=cols,
                    is_keyframe=is_kf, frame_index=frame_idx,
                )
        finally:
            cap.release()

    # ── Point cloud access ───────────────────────────────────────────

    @property
    def points(self):
        """Current accumulated points as Nx3 float32 array."""
        pts, _ = self._cloud.get_data()
        return pts

    @property
    def colors(self):
        """Current accumulated colors as Nx3 uint8 array (RGB)."""
        _, cols = self._cloud.get_data()
        return cols

    @property
    def point_count(self):
        """Number of accumulated points."""
        return self._cloud.count()

    def save(self, filepath):
        """Save current point cloud to a PLY file."""
        pts, cols = self._cloud.get_data()
        save_ply(filepath, pts, cols)

    def load(self, filepath):
        """Load a PLY file, replacing the current point cloud."""
        pts, cols = load_ply(filepath)
        with self._cloud.lock:
            self._cloud.points = pts
            self._cloud.colors = cols

    def reset(self):
        """Clear all accumulated data and reset pose."""
        with self._cloud.lock:
            self._cloud.points = np.zeros((0, 3), dtype=np.float32)
            self._cloud.colors = np.zeros((0, 3), dtype=np.uint8)
        self._last_kf_pose = np.eye(4, dtype=np.float64)
        self._frames_since_kf = 0
        self._keyframe_count = 0
        if self._pose_est:
            self._pose_est.pose = np.eye(4, dtype=np.float64)
            self._pose_est.prev_pts = None


# ============================================================================
# 9. DEMO VIDEO RENDERER
# ============================================================================
import math


def _render_pointcloud_view(points, colors, frame_idx, total_frames,
                            width=640, height=480):
    """Render the 3D point cloud with depth shading and gentle Z-axis b-roll."""
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    if len(points) < 10:
        cv2.putText(canvas, "Accumulating points...",
                    (width // 6, height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
        return canvas

    centroid = points.mean(axis=0)
    pts = points - centroid
    spread = np.percentile(np.abs(pts), 95)
    if spread < 1e-6:
        spread = 1.0

    angle = (frame_idx / max(total_frames, 1)) * math.pi * 0.15
    cos_a, sin_a = math.cos(angle), math.sin(angle)

    rx = pts[:, 0] * cos_a - pts[:, 1] * sin_a
    ry = pts[:, 0] * sin_a + pts[:, 1] * cos_a
    rz = pts[:, 2]

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

    z_min, z_max = z_shifted.min(), z_shifted.max()
    z_range = z_max - z_min + 1e-6
    depth_factor = 1.0 - 0.6 * (z_shifted - z_min) / z_range
    cols_shaded = cols_valid * depth_factor[:, np.newaxis]
    cols_shaded = np.clip(cols_shaded, 0, 255).astype(np.uint8)

    depth_order = np.argsort(-z_shifted)
    px, py = px[depth_order], py[depth_order]
    cols_shaded = cols_shaded[depth_order]

    mask = (px >= 1) & (px < width - 1) & (py >= 1) & (py < height - 1)
    px, py = px[mask], py[mask]
    cols_shaded = cols_shaded[mask]

    if len(px) > 250000:
        step = max(1, len(px) // 250000)
        px, py, cols_shaded = px[::step], py[::step], cols_shaded[::step]

    bgr = cols_shaded[:, ::-1]
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            canvas[py + dy, px + dx] = bgr

    return canvas


def _add_label(panel, label, bg_color=(0, 0, 0)):
    """Add a label bar at the top of a panel."""
    bar = np.full((32, panel.shape[1], 3), bg_color, dtype=np.uint8)
    cv2.putText(bar, label, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2)
    return np.vstack([bar, panel[:panel.shape[0] - 32]])


def render_demo_video(video_path, output_path=None):
    """Render a 4-panel demo video: Camera | Depth | Features | 3D Point Cloud."""
    _init_runtime()

    video_path = str(video_path)
    if output_path is None:
        out_dir = Path.cwd() / "output_video"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / "eridian_demo.mp4"

    PANEL_W, PANEL_H = 640, 480

    log.info("=" * 60)
    log.info("Rendering 4-panel demo video")
    log.info("=" * 60)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log.error(f"Cannot open {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    in_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    log.info(f"Input: {video_path}, {total_frames} frames @ {in_fps:.0f}fps")

    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(str(output_path), fourcc, in_fps, (PANEL_W * 2, PANEL_H * 2))
    if not out.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, in_fps, (PANEL_W * 2, PANEL_H * 2))

    log.info("Loading depth model...")
    depth_est = DepthEstimator()

    K_fx = PANEL_W * 0.8
    K = np.array([[K_fx, 0, PANEL_W / 2],
                  [0, K_fx, PANEL_H / 2],
                  [0, 0, 1]], dtype=np.float64)
    pose_est = PoseEstimator(K_fx, K_fx, PANEL_W / 2, PANEL_H / 2)
    cloud = PointCloud()

    last_kf_pose = np.eye(4)
    frames_since_kf = 0
    frame_idx = 0
    t_start = time.time()

    cached_pts = np.zeros((0, 3), dtype=np.float32)
    cached_cols = np.zeros((0, 3), dtype=np.uint8)

    while True:
        ok, raw_frame = cap.read()
        if not ok:
            break

        frame = cv2.resize(raw_frame, (PANEL_W, PANEL_H))
        frame_idx += 1

        try:
            depth = depth_est.estimate(frame)
        except Exception as e:
            log.warning(f"Frame {frame_idx}: depth failed: {e}")
            depth = np.ones((PANEL_H, PANEL_W), dtype=np.float32) * 5.0

        try:
            pose, kp, matches_img = pose_est.update(frame, depth=depth)
        except Exception:
            pose = np.eye(4)
            matches_img = frame.copy()

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

        elapsed = time.time() - t_start
        fps = frame_idx / elapsed if elapsed > 0 else 0

        p1 = frame.copy()
        cv2.putText(p1, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(p1, f"Frame: {frame_idx}/{total_frames}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        p1 = _add_label(p1, "CAMERA")

        d_norm = ((depth - depth.min()) / (depth.max() - depth.min() + 1e-6) * 255).astype(np.uint8)
        p2 = cv2.applyColorMap(d_norm, cv2.COLORMAP_INFERNO)
        p2 = _add_label(p2, "DEPTH (metric)")

        p3 = matches_img.copy()
        cv2.putText(p3, f"Tracking: {'OK' if pose_est.tracking else 'LOST'}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0) if pose_est.tracking else (0, 0, 255), 2)
        p3 = _add_label(p3, "OPTICAL FLOW + PnP")

        if frame_idx % 5 == 1 or frame_idx == 1:
            cached_pts, cached_cols = cloud.get_data()
        p4 = _render_pointcloud_view(cached_pts, cached_cols, frame_idx, total_frames)
        cv2.putText(p4, f"Points: {len(cached_pts):,}", (10, PANEL_H - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        p4 = _add_label(p4, "3D POINT CLOUD")

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
    file_size = Path(output_path).stat().st_size / (1024 * 1024)
    log.info("=" * 60)
    log.info(f"Done! {frame_idx} frames in {duration:.1f}s")
    log.info(f"Output: {output_path} ({file_size:.1f} MB)")
    log.info("=" * 60)
    return str(output_path)


# ============================================================================
# ENTRY POINT
# ============================================================================
def cli_main():
    """Entry point for both `python main.py` and the `eridian` console script."""
    _init_runtime()
    parser = argparse.ArgumentParser(description="Eridian - 3D World Mapper")
    parser.add_argument("--test", action="store_true",
                        help="Use test video (./data/video.mp4) instead of camera")
    parser.add_argument("--video", type=str, default=None,
                        help="Path to a video file to use as input")
    parser.add_argument("--camera", type=int, default=None,
                        help="Camera index to use (skip interactive picker)")
    parser.add_argument("--headless", action="store_true",
                        help="Run without GUI windows (process and save only)")
    parser.add_argument("--save", action="store_true",
                        help="Save PLY point cloud to current directory")
    parser.add_argument("--render", action="store_true",
                        help="Render a 4-panel demo video from input video")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for rendered video (used with --render)")
    args = parser.parse_args()

    video = args.video
    headless = args.headless
    camera_index = args.camera

    if args.test:
        video = str(TEST_VIDEO)
        headless = True

    if args.render:
        src = video or str(TEST_VIDEO)
        render_demo_video(src, args.output)
        return

    # Interactive camera picker when no video and no camera specified
    if not video and camera_index is None:
        print("Scanning for cameras...")
        cameras = probe_cameras()
        camera_index = pick_camera(cameras)
        if camera_index is None:
            print("No camera selected. Use --video or --test instead.")
            return

    mapper = WorldMapper(video_path=video, headless=headless,
                         camera_index=camera_index, save=args.save)
    mapper.run()


if __name__ == "__main__":
    cli_main()
