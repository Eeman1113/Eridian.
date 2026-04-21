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


# ── Depth model presets ───────────────────────────────────────────────────────
DEPTH_MODELS = {
    'small': "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
    'base':  "depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf",
    'large': "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
}


# ============================================================================
# CONFIGURATION
# ============================================================================
@dataclasses.dataclass
class EridianConfig:
    """Tunable parameters for quality/speed tradeoff."""

    # Depth resolution (294 = 14*21 ViT patch-aligned: best MAE at real-time speed)
    fast_resolution: int = 294
    quality_resolution: int = 518

    # Tracking
    max_corners: int = 800
    fb_threshold: float = 0.5        # forward-backward consistency (px)
    pnp_reproj_error: float = 2.0    # PnP RANSAC reprojection threshold (px)
    pnp_iterations: int = 300
    use_velocity_prior: bool = True
    pnp_refine: bool = True          # LM refinement on inliers
    subpixel_corners: bool = True

    # Keyframes
    kf_translation: float = 0.12     # min translation for keyframe (m)
    kf_rotation: float = 5.0         # min rotation for keyframe (degrees)
    kf_timeout: int = 30             # max frames between keyframes
    kf_min_inlier_ratio: float = 0.4 # reject keyframes with bad pose

    # Point cloud
    voxel_size: float = 0.015
    max_points: int = 5_000_000
    subsample: int = 2
    confidence_keep: float = 0.8

    # Depth filtering
    fast_bilateral: bool = True      # light bilateral on tracking depth


_DEFAULT_CFG = EridianConfig()

# ── Processor resolution presets (pixels) ─────────────────────────────────────
_PROC_FAST = {'height': _DEFAULT_CFG.fast_resolution,
              'width': _DEFAULT_CFG.fast_resolution}
_PROC_QUALITY = {'height': _DEFAULT_CFG.quality_resolution,
                 'width': _DEFAULT_CFG.quality_resolution}


# ============================================================================
# 2. MONOCULAR DEPTH ESTIMATION (multi-model, consistency smoothing)
# ============================================================================
class DepthEstimator:
    """Depth estimation with model selection: small/base/large (DAv2 Metric)."""

    def __init__(self, model='small', config=None):
        self.model_choice = model
        self.cfg = config or _DEFAULT_CFG
        self.pipe = None
        self._model = None
        self._processor = None
        self.model_name = None
        self.is_metric = False
        self.prev_depth = None
        self._device = None
        self._proc_fast = {'height': self.cfg.fast_resolution,
                           'width': self.cfg.fast_resolution}
        self._proc_quality = {'height': self.cfg.quality_resolution,
                              'width': self.cfg.quality_resolution}
        self._load_model()

    def _load_model(self):
        self._device = "mps" if torch.backends.mps.is_available() else "cpu"
        self._load_pipeline()

    def _load_pipeline(self):
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        # Build candidate list: selected model first, then fallbacks
        selected = DEPTH_MODELS.get(self.model_choice, DEPTH_MODELS['small'])
        candidates = [(selected, True)]
        for key, name in DEPTH_MODELS.items():
            if name != selected:
                candidates.append((name, True))
        candidates.extend([
            ("depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf", True),
            ("depth-anything/Depth-Anything-V2-Small-hf", False),
            ("Intel/dpt-swinv2-tiny-256", False),
            ("Intel/dpt-hybrid-midas", False),
        ])

        for name, metric in candidates:
            try:
                log.info(f"Loading depth model: {name} (float16, channels_last)")
                self._processor = AutoImageProcessor.from_pretrained(name)
                self._model = AutoModelForDepthEstimation.from_pretrained(
                    name, torch_dtype=torch.float16,
                )
                self._model = self._model.to(self._device).eval()
                try:
                    self._model = self._model.to(memory_format=torch.channels_last)
                except Exception:
                    pass  # not all architectures support channels_last
                self.model_name = name
                self.is_metric = metric
                self.pipe = None
                if self._device == "mps":
                    torch.mps.empty_cache()
                log.info(f"Depth model loaded: {name} ({'METRIC' if metric else 'relative'}, fp16)")
                return
            except Exception as e:
                log.warning(f"Failed to load {name}: {e}")

        raise RuntimeError("Could not load any depth estimation model")

    # ── Inference ─────────────────────────────────────────────────────────────

    def estimate(self, frame_bgr):
        """Quality depth at full resolution. Returns float32 HxW in metres."""
        return self._estimate(frame_bgr, fast=False)

    def estimate_fast(self, frame_bgr):
        """Fast depth at reduced resolution for tracking (~5x faster)."""
        return self._estimate(frame_bgr, fast=True)

    def _estimate(self, frame_bgr, fast=False):
        from PIL import Image as PILImage
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = PILImage.fromarray(rgb)
        h, w = frame_bgr.shape[:2]

        proc_size = self._proc_fast if fast else self._proc_quality
        inputs = self._processor(images=pil_img, size=proc_size, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(
            self._device, dtype=torch.float16,
            memory_format=torch.channels_last,
        )

        with torch.inference_mode():
            out = self._model(pixel_values)

        depth = out.predicted_depth
        # Resize on GPU before transfer
        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(0).float(), size=(h, w),
            mode="bilinear", align_corners=False,
        ).squeeze().cpu().numpy().astype(np.float32)

        if not fast:
            depth = cv2.bilateralFilter(depth, d=5, sigmaColor=0.2, sigmaSpace=5)
            return self._consistency_smooth(depth)
        # Light bilateral on fast path: reduces PnP noise (~2ms cost)
        if self.cfg.fast_bilateral:
            depth = cv2.bilateralFilter(depth, d=3, sigmaColor=0.15, sigmaSpace=3)
        return depth

    def _consistency_smooth(self, depth):
        """Smooth only where temporally consistent — no blending at depth edges."""
        if self.prev_depth is not None and self.prev_depth.shape == depth.shape:
            diff = np.abs(depth - self.prev_depth)
            consistent = diff < (0.1 * depth)
            smoothed = 0.8 * depth + 0.2 * self.prev_depth
            depth = np.where(consistent, smoothed, depth)
        self.prev_depth = depth.copy()
        return depth


# ============================================================================
# 3. POSE ESTIMATION (GFTT + Lucas-Kanade + PnP)
# ============================================================================
class PoseEstimator:
    """Visual odometry: GFTT + Lucas-Kanade + PnP with motion model."""

    def __init__(self, fx, fy, cx, cy, config=None):
        self.cfg = config or _DEFAULT_CFG
        self.K = np.array([[fx, 0, cx],
                           [0, fy, cy],
                           [0,  0,  1]], dtype=np.float64)
        self.fx, self.fy = fx, fy
        self.cx, self.cy = cx, cy

        self.gftt_params = dict(
            maxCorners=self.cfg.max_corners, qualityLevel=0.005,
            minDistance=10, blockSize=7,
        )

        # Lucas-Kanade optical flow params
        self.lk_params = dict(
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )

        # Subpixel refinement criteria
        self._subpix_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01)

        self.prev_gray = None
        self.prev_pts = None
        self.prev_depth = None
        self.pose = np.eye(4, dtype=np.float64)
        self.tracking = True
        self.inlier_ratio = 0.0
        self.frame_count = 0
        # Constant-velocity motion model
        self._prev_pose = np.eye(4, dtype=np.float64)
        self._velocity = np.eye(4, dtype=np.float64)

    def update_intrinsics(self, fx, fy, cx, cy):
        """Update camera intrinsics at runtime."""
        self.K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        self.fx, self.fy = fx, fy
        self.cx, self.cy = cx, cy

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

            # Subpixel corner refinement (~1ms, much better flow init)
            if self.cfg.subpixel_corners and pts is not None and len(pts) > 0:
                pts = cv2.cornerSubPix(gray, pts, (5, 5), (-1, -1),
                                       self._subpix_criteria)

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
            # Keep only points that survived both directions with tight round-trip error
            good_mask = status & status_back & (fb_err < self.cfg.fb_threshold)
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
            self.inlier_ratio = 0.0
            self.prev_gray = gray
            self.prev_pts = self._detect_corners(gray)
            self.prev_depth = depth
            return self.pose.copy(), [], viz

        # ── PnP pose from depth ────────────────────────────────────
        if self.prev_depth is not None:
            # Bilinear depth interpolation (much more accurate than nearest-neighbor)
            map_x = prev_good[:, 0:1].astype(np.float32)
            map_y = prev_good[:, 1:2].astype(np.float32)
            z = cv2.remap(self.prev_depth, map_x, map_y,
                          cv2.INTER_LINEAR).ravel()

            # Filter valid depth
            valid = (z > 0.2) & (z < 12.0)

            # Reject depth-edge features: depth must be consistent with 3x3 neighborhood
            if valid.sum() >= 8:
                px_i = np.clip(prev_good[:, 0].astype(int), 1,
                               self.prev_depth.shape[1] - 2)
                py_i = np.clip(prev_good[:, 1].astype(int), 1,
                               self.prev_depth.shape[0] - 2)
                local_med = np.median(
                    np.stack([self.prev_depth[py_i - 1, px_i],
                              self.prev_depth[py_i + 1, px_i],
                              self.prev_depth[py_i, px_i - 1],
                              self.prev_depth[py_i, px_i + 1],
                              self.prev_depth[py_i, px_i]], axis=0), axis=0)
                depth_consistent = np.abs(z - local_med) < 0.10 * z
                valid &= depth_consistent

            if valid.sum() < 8:
                self.tracking = False
                self.inlier_ratio = 0.0
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

            # Velocity-model prior as PnP initial guess
            pnp_kwargs = dict(
                iterationsCount=self.cfg.pnp_iterations,
                reprojectionError=self.cfg.pnp_reproj_error,
                flags=cv2.SOLVEPNP_AP3P,
            )
            if self.cfg.use_velocity_prior and self.frame_count > 2:
                predicted = self._velocity @ self.pose
                T_pred = np.linalg.inv(predicted) @ self.pose
                rvec_init, _ = cv2.Rodrigues(T_pred[:3, :3])
                tvec_init = T_pred[:3, 3].reshape(3, 1)
                pnp_kwargs.update(rvec=rvec_init, tvec=tvec_init,
                                  useExtrinsicGuess=True)

            ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                pts_3d, pts_2d, self.K, None, **pnp_kwargs)

            n_total = len(pts_3d)
            n_inliers = 0 if inliers is None else len(inliers)
            self.inlier_ratio = n_inliers / n_total if n_total > 0 else 0.0

            if ok and inliers is not None and len(inliers) >= 8:
                # Refine PnP with Levenberg-Marquardt on inliers only
                if self.cfg.pnp_refine and len(inliers) >= 10:
                    inl = inliers.ravel()
                    ok2, rvec, tvec = cv2.solvePnP(
                        pts_3d[inl], pts_2d[inl], self.K, None,
                        rvec=rvec, tvec=tvec, useExtrinsicGuess=True,
                        flags=cv2.SOLVEPNP_ITERATIVE,
                    )

                R, _ = cv2.Rodrigues(rvec)
                T_rel = np.eye(4, dtype=np.float64)
                T_rel[:3, :3] = R
                T_rel[:3, 3] = tvec.ravel()

                new_pose = self.pose @ np.linalg.inv(T_rel)

                # Sanity check: reject poses with NaN/Inf or unreasonable jumps
                t_jump = np.linalg.norm(new_pose[:3, 3] - self.pose[:3, 3])
                if np.isnan(new_pose).any() or np.isinf(new_pose).any() or t_jump > 5.0:
                    log.warning(f"PnP produced bad pose (jump={t_jump:.2f}m), holding position")
                    self.tracking = False
                else:
                    # Update constant-velocity motion model
                    self._velocity = new_pose @ np.linalg.inv(self.pose)
                    self._prev_pose = self.pose.copy()
                    self.pose = new_pose
                    self.tracking = True

                # Highlight inliers
                for idx in inliers.ravel():
                    if idx < len(curr_2d):
                        x, y = int(curr_2d[idx, 0]), int(curr_2d[idx, 1])
                        cv2.circle(viz, (x, y), 4, (255, 0, 255), 1)
            else:
                log.warning(f"PnP failed ({n_inliers} inliers / {n_total} pts), holding position")
                self.tracking = False

        # ── Update state ───────────────────────────────────────────
        # Re-detect if too few tracked points
        if len(curr_good) < 100:
            self.prev_pts = self._detect_corners(gray)
        else:
            self.prev_pts = curr_good.reshape(-1, 1, 2).astype(np.float32)
        self.prev_gray = gray
        self.prev_depth = depth

        return self.pose.copy(), [], viz

    def _detect_corners(self, gray):
        """Detect GFTT corners with optional subpixel refinement."""
        pts = cv2.goodFeaturesToTrack(gray, **self.gftt_params)
        if pts is not None and self.cfg.subpixel_corners and len(pts) > 0:
            pts = cv2.cornerSubPix(gray, pts, (5, 5), (-1, -1),
                                   self._subpix_criteria)
        return pts


# ============================================================================
# 4. POINT CLOUD (with edge/normal filtering + voxel averaging)
# ============================================================================
class PointCloud:
    """Global point cloud: dense sampling, confidence weighting, adaptive voxels, outlier removal."""

    def __init__(self, max_points=5_000_000, voxel_size=0.015):
        self.points = np.zeros((0, 3), dtype=np.float32)
        self.colors = np.zeros((0, 3), dtype=np.uint8)
        self.normals = np.zeros((0, 3), dtype=np.float32)
        self.max_points = max_points
        self.voxel_size = voxel_size
        self.lock = Lock()
        self.frame_count = 0

    def add_frame(self, depth, frame_bgr, pose, K, subsample=2):
        """Back-project depth to 3D with quality filtering, confidence weighting, normals."""
        # Guard: reject poses with NaN/Inf
        if np.isnan(pose).any() or np.isinf(pose).any():
            return

        h, w = depth.shape
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        # ── Depth gradients (for edge + normal filtering) ──────────
        grad_x = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

        # ── Subsample grid (2x denser than before) ─────────────────
        ys = np.arange(0, h, subsample)
        xs = np.arange(0, w, subsample)
        xs, ys = np.meshgrid(xs, ys)
        xs, ys = xs.ravel(), ys.ravel()

        z = depth[ys, xs]
        gm = grad_mag[ys, xs]

        # ── Filter 1: valid depth range ────────────────────────────
        valid = (z > 0.15) & (z < 15.0)

        # ── Filter 2: depth edge rejection (tighter threshold) ─────
        edge_thresh = 0.10 * z
        valid &= (gm < edge_thresh)

        # ── Filter 3: grazing angle rejection via surface normals ──
        gx = grad_x[ys, xs]
        gy = grad_y[ys, xs]
        nx = -gx / fx
        ny = -gy / fy
        nz = np.ones_like(nx)
        n_norm = np.sqrt(nx ** 2 + ny ** 2 + nz ** 2) + 1e-8
        nx, ny, nz = nx / n_norm, ny / n_norm, nz / n_norm

        ray_x = (xs.astype(np.float32) - cx) / fx
        ray_y = (ys.astype(np.float32) - cy) / fy
        ray_z = np.ones_like(ray_x)
        ray_norm = np.sqrt(ray_x ** 2 + ray_y ** 2 + ray_z ** 2)
        ray_x, ray_y, ray_z = ray_x / ray_norm, ray_y / ray_norm, ray_z / ray_norm

        cos_angle = np.abs(nx * ray_x + ny * ray_y + nz * ray_z)
        valid &= (cos_angle > 0.3)  # reject angles > ~72 degrees

        # ── Apply filter ───────────────────────────────────────────
        xs, ys, z = xs[valid], ys[valid], z[valid]
        cos_angle = cos_angle[valid]
        nx_v, ny_v, nz_v = nx[valid], ny[valid], nz[valid]

        if len(z) < 10:
            return

        # ── Confidence-weighted sampling ───────────────────────────
        # Points closer + viewed more head-on are more accurate
        confidence = cos_angle / (z * z + 1e-6)
        conf_norm = confidence / (confidence.max() + 1e-8)
        # Keep top 80% by confidence (drop worst 20%)
        keep_n = max(10, int(len(z) * 0.8))
        if len(z) > keep_n:
            top_idx = np.argpartition(conf_norm, -keep_n)[-keep_n:]
            xs, ys, z = xs[top_idx], ys[top_idx], z[top_idx]
            nx_v, ny_v, nz_v = nx_v[top_idx], ny_v[top_idx], nz_v[top_idx]

        # ── Back-project to 3D camera coords ──────────────────────
        x3d = (xs.astype(np.float32) - cx) * z / fx
        y3d = (ys.astype(np.float32) - cy) * z / fy
        pts_cam = np.stack([x3d, y3d, z], axis=1)

        # ── Camera-frame normals ───────────────────────────────────
        normals_cam = np.stack([nx_v, ny_v, nz_v], axis=1)

        # ── Transform to world coordinates ─────────────────────────
        R = pose[:3, :3].astype(np.float64)
        t = pose[:3, 3].astype(np.float64)
        pts_world = (pts_cam.astype(np.float64) @ R.T + t).astype(np.float32)
        normals_world = (normals_cam.astype(np.float64) @ R.T).astype(np.float32)

        # ── Filter out any NaN/Inf points ─────────────────────────
        finite_mask = np.isfinite(pts_world).all(axis=1)
        if not finite_mask.all():
            pts_world = pts_world[finite_mask]
            normals_world = normals_world[finite_mask]
            xs, ys = xs[finite_mask], ys[finite_mask]
            if len(pts_world) < 10:
                return

        # ── Colors (BGR -> RGB) ────────────────────────────────────
        colors = frame_bgr[ys, xs][:, ::-1].copy()

        with self.lock:
            self.points = np.vstack([self.points, pts_world])
            self.colors = np.vstack([self.colors, colors])
            self.normals = np.vstack([self.normals, normals_world])
            self.frame_count += 1

            if self.frame_count % 20 == 0:
                self._voxel_downsample()

    def _voxel_downsample(self):
        """Voxel averaging with outlier removal: low-observation voxels are dropped."""
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

        counts = np.bincount(inverse, minlength=n_voxels).astype(np.float32)
        avg_pts = np.zeros((n_voxels, 3), dtype=np.float32)
        avg_cols = np.zeros((n_voxels, 3), dtype=np.float32)
        avg_norms = np.zeros((n_voxels, 3), dtype=np.float32)
        for dim in range(3):
            avg_pts[:, dim] = np.bincount(
                inverse, weights=self.points[:, dim], minlength=n_voxels
            ) / counts
            avg_cols[:, dim] = np.bincount(
                inverse, weights=self.colors[:, dim].astype(np.float32), minlength=n_voxels
            ) / counts
            avg_norms[:, dim] = np.bincount(
                inverse, weights=self.normals[:, dim], minlength=n_voxels
            ) / counts

        # Normalize averaged normals
        nrm = np.linalg.norm(avg_norms, axis=1, keepdims=True) + 1e-8
        avg_norms = avg_norms / nrm

        # ── Statistical outlier removal: drop voxels with only 1 observation ──
        reliable = counts >= 2
        avg_pts = avg_pts[reliable]
        avg_cols = avg_cols[reliable]
        avg_norms = avg_norms[reliable]
        counts = counts[reliable]

        self.points = avg_pts
        self.colors = np.clip(avg_cols, 0, 255).astype(np.uint8)
        self.normals = avg_norms

        if len(self.points) > self.max_points:
            keep = np.argsort(-counts)[:self.max_points]
            self.points = self.points[keep]
            self.colors = self.colors[keep]
            self.normals = self.normals[keep]

        log.info(f"Voxel average: {n_before} -> {len(self.points)} points")

    def get_data(self):
        with self.lock:
            return self.points.copy(), self.colors.copy()

    def get_data_with_normals(self):
        with self.lock:
            return self.points.copy(), self.colors.copy(), self.normals.copy()

    def count(self):
        return len(self.points)


# ============================================================================
# 5. PLY EXPORT
# ============================================================================
def save_ply(filepath, points, colors, normals=None):
    """Save point cloud as binary PLY (camera POV: Y-up), optionally with normals."""
    n = len(points)
    if n == 0:
        log.warning("No points to save")
        return

    # Flip Y axis so the PLY is right-side-up (camera convention is Y-down)
    points = points.copy()
    points[:, 1] = -points[:, 1]

    has_normals = normals is not None and len(normals) == n
    if has_normals:
        normals = normals.copy()
        normals[:, 1] = -normals[:, 1]

    props = (
        "property float x\n"
        "property float y\n"
        "property float z\n"
    )
    if has_normals:
        props += (
            "property float nx\n"
            "property float ny\n"
            "property float nz\n"
        )
    props += (
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
    )

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        f"{props}"
        "end_header\n"
    )

    if has_normals:
        dtype = np.dtype([
            ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
            ('nx', '<f4'), ('ny', '<f4'), ('nz', '<f4'),
            ('r', 'u1'), ('g', 'u1'), ('b', 'u1'),
        ])
    else:
        dtype = np.dtype([
            ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
            ('r', 'u1'), ('g', 'u1'), ('b', 'u1'),
        ])

    data = np.empty(n, dtype=dtype)
    data['x'] = points[:, 0]
    data['y'] = points[:, 1]
    data['z'] = points[:, 2]
    if has_normals:
        data['nx'] = normals[:, 0]
        data['ny'] = normals[:, 1]
        data['nz'] = normals[:, 2]
    data['r'] = colors[:, 0]
    data['g'] = colors[:, 1]
    data['b'] = colors[:, 2]

    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
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

    def __init__(self, video_path=None, headless=False, camera_index=None, save=False,
                 savedir=None, model='small', config=None):
        self.cfg = config or _DEFAULT_CFG
        self.running = True
        self.camera = None
        self.depth_estimator = None
        self.pose_estimator = None
        self.cloud = PointCloud(max_points=self.cfg.max_points,
                                voxel_size=self.cfg.voxel_size)
        self.viz3d = Visualizer3D()
        self.video_path = video_path
        self.headless = headless
        self.camera_index = camera_index
        self.save_to_cwd = save
        self.savedir = Path(savedir) if savedir else None
        self.model_choice = model

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

        # Gate: reject keyframes when tracking is unreliable
        if not self.pose_estimator.tracking:
            return False
        if self.pose_estimator.inlier_ratio < self.cfg.kf_min_inlier_ratio:
            return False

        t_diff = np.linalg.norm(pose[:3, 3] - self.last_kf_pose[:3, 3])
        R_diff = pose[:3, :3] @ self.last_kf_pose[:3, :3].T
        angle = np.arccos(np.clip((np.trace(R_diff) - 1.0) / 2.0, -1.0, 1.0))

        if (t_diff > self.cfg.kf_translation
                or angle > np.radians(self.cfg.kf_rotation)
                or self.frames_since_kf > self.cfg.kf_timeout):
            self.last_kf_pose = pose.copy()
            self.frames_since_kf = 0
            self.keyframe_count += 1
            return True
        return False

    def _save_state(self, tag=""):
        pts, cols, norms = self.cloud.get_data_with_normals()
        if len(pts) == 0:
            return
        out = self.savedir if self.savedir else SPLAT_DIR
        out.mkdir(parents=True, exist_ok=True)
        save_ply(out / "cloud_latest.ply", pts, cols, norms)
        if tag:
            save_ply(out / f"cloud_{tag}.ply", pts, cols, norms)
        if self.save_to_cwd:
            save_ply(Path.cwd() / "eridian_cloud.ply", pts, cols, norms)

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

        # ── Init depth model ───────────────────────────────────────
        log.info(f"Loading {self.model_choice} (fp16, "
                 f"track@{self.cfg.fast_resolution} / cloud@{self.cfg.quality_resolution})...")
        try:
            self.depth_estimator = DepthEstimator(model=self.model_choice,
                                                  config=self.cfg)
        except Exception as e:
            log.error(f"Failed to load depth model: {e}")
            return

        # ── Camera intrinsics ──────────────────────────────────────
        fx = fy = w * 0.8
        cx, cy = w / 2.0, h / 2.0
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        log.info(f"Camera intrinsics: fx={fx:.0f} fy={fy:.0f} cx={cx:.0f} cy={cy:.0f}")

        self.pose_estimator = PoseEstimator(fx, fy, cx, cy, config=self.cfg)

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

                # ── Depth (every frame for accurate tracking) ──────
                depth = None
                try:
                    depth = self.depth_estimator.estimate_fast(frame)
                except Exception as e:
                    log.warning(f"Depth estimation failed: {e}")

                # ── Pose ───────────────────────────────────────────
                pose = None
                matches_img = frame.copy()
                try:
                    pose, kp, matches_img = self.pose_estimator.update(frame, depth=depth)
                except Exception as e:
                    log.warning(f"Pose estimation failed: {e}")

                # ── Keyframe: quality depth for cloud ──────────────
                display_depth = depth  # default: show fast depth
                if depth is not None and pose is not None:
                    if self._is_keyframe(pose):
                        try:
                            quality_depth = self.depth_estimator.estimate(frame)
                            self.cloud.add_frame(quality_depth, frame, pose, K)
                            display_depth = quality_depth  # show sharp depth on keyframes
                        except Exception as e:
                            log.warning(f"Point cloud accumulation failed: {e}")

                # ── FPS (sync MPS for accurate timing) ────────────
                if torch.backends.mps.is_available():
                    torch.mps.synchronize()
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

                    if display_depth is not None:
                        # Percentile normalization for better contrast in narrow-range scenes
                        d_lo = np.percentile(display_depth, 2)
                        d_hi = np.percentile(display_depth, 98)
                        d_norm = np.clip((display_depth - d_lo) / (d_hi - d_lo + 1e-6), 0, 1)
                        d_vis = (d_norm * 255).astype(np.uint8)
                        cv2.imshow("Depth", cv2.applyColorMap(d_vis, cv2.COLORMAP_TURBO))

                    cv2.imshow("Features", matches_img)

                    if self.frame_idx % 5 == 0:
                        pts, cols = self.cloud.get_data()
                        self.viz3d.update(pts, cols)

                # ── Save depth frames ──────────────────────────────
                if depth is not None and self.frame_idx % 15 == 0:
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

    def __init__(self, model='small'):
        self._model_choice = model
        self._depth_est = None
        self._pose_est = None
        self._cloud = PointCloud()
        self._K = None
        self._last_kf_pose = np.eye(4, dtype=np.float64)
        self._frames_since_kf = 0
        self._keyframe_count = 0
        self._callbacks = {"on_frame": [], "on_keyframe": [], "on_depth": []}
        self._intrinsics_set = False

    def _ensure_depth(self):
        if self._depth_est is None:
            self._depth_est = DepthEstimator(model=self._model_choice)

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


def render_demo_video(video_path, output_path=None, model='small'):
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

    log.info(f"Loading depth model ({model})...")
    depth_est = DepthEstimator(model=model)
    tracking_est = depth_est
    keyframe_est = depth_est

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

        # Fast depth for tracking, quality depth for keyframe cloud
        try:
            depth = tracking_est.estimate_fast(frame)
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
                quality_depth = keyframe_est.estimate(frame)
                cloud.add_frame(quality_depth, frame, pose, K)
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
def _run_gct_mode(args):
    """Run GCT-based 3D reconstruction (LingBot-MAP transformer model)."""
    from eridian.gct_backend import GCTReconstructor, gct_to_pointcloud

    video = args.video
    if args.test:
        video = str(TEST_VIDEO)

    if not video:
        print("GCT mode requires --video or --test (camera streaming not yet supported).")
        print("Usage: eridian --gct --model_path /path/to/checkpoint.pt --video video.mp4")
        return

    if not args.gct_model:
        print("GCT mode requires --gct-model /path/to/checkpoint.pt")
        print("Download from: https://github.com/Robbyant/lingbot-map")
        return

    print("=" * 60)
    print("Eridian + GCT (LingBot-MAP)")
    print("3D Reconstruction powered by Geometric Context Transformer")
    print("Model: https://github.com/Robbyant/lingbot-map (Apache 2.0)")
    print("=" * 60)

    t0 = time.time()
    gct = GCTReconstructor(
        model_path=args.gct_model,
        image_size=args.gct_image_size,
        num_scale_frames=args.gct_scale_frames,
        conf_threshold=args.gct_conf,
        kv_cache_sliding_window=args.gct_window,
    )

    result = gct.reconstruct_video(
        video,
        fps=args.gct_fps,
        max_frames=args.gct_max_frames,
        keyframe_interval=args.gct_keyframe_interval,
    )

    points, colors = gct_to_pointcloud(
        result,
        conf_threshold=args.gct_conf,
        subsample=args.gct_subsample,
    )

    elapsed = time.time() - t0
    print(f"\nReconstruction complete: {result['num_frames']} frames in {elapsed:.1f}s")
    print(f"Points: {len(points):,}")

    if len(points) > 0:
        out_dir = Path(args.savedir) if args.savedir else SPLAT_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "gct_cloud.ply"
        save_ply(out_path, points, colors)
        print(f"Saved: {out_path}")

        if args.save:
            cwd_path = Path.cwd() / "eridian_gct_cloud.ply"
            save_ply(cwd_path, points, colors)
            print(f"Saved: {cwd_path}")
    else:
        print("No points passed confidence threshold. Try lowering --gct-conf.")


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
    parser.add_argument("--savedir", type=str, default=None,
                        help="Directory to save PLY point clouds (default: ./splat/)")
    parser.add_argument("--model", type=str, default="small",
                        choices=["small", "base", "large"],
                        help="Depth model: small (fast), base (balanced), large (highest quality)")

    # GCT mode (LingBot-MAP transformer)
    gct_group = parser.add_argument_group(
        "GCT mode (LingBot-MAP)",
        "Use the Geometric Context Transformer for 3D reconstruction. "
        "Credit: https://github.com/Robbyant/lingbot-map (Apache 2.0)"
    )
    gct_group.add_argument("--gct", action="store_true",
                           help="Use GCT transformer model instead of depth+pose pipeline")
    gct_group.add_argument("--gct-model", type=str, default=None,
                           help="Path to GCT checkpoint (.pt file)")
    gct_group.add_argument("--gct-fps", type=int, default=10,
                           help="Frame sampling rate for video (default: 10)")
    gct_group.add_argument("--gct-max-frames", type=int, default=None,
                           help="Max frames to process")
    gct_group.add_argument("--gct-scale-frames", type=int, default=8,
                           help="Number of scale estimation frames (default: 8)")
    gct_group.add_argument("--gct-window", type=int, default=64,
                           help="KV cache sliding window size (default: 64)")
    gct_group.add_argument("--gct-conf", type=float, default=1.5,
                           help="Confidence threshold for point filtering (default: 1.5)")
    gct_group.add_argument("--gct-subsample", type=int, default=4,
                           help="Spatial subsampling factor (default: 4)")
    gct_group.add_argument("--gct-image-size", type=int, default=518,
                           help="Input image size for GCT (default: 518)")
    gct_group.add_argument("--gct-keyframe-interval", type=int, default=None,
                           help="KV cache keyframe interval (auto if unset)")

    args = parser.parse_args()

    video = args.video
    headless = args.headless
    camera_index = args.camera

    if args.test:
        video = str(TEST_VIDEO)
        headless = True

    # GCT mode
    if args.gct:
        _run_gct_mode(args)
        return

    if args.render:
        src = video or str(TEST_VIDEO)
        render_demo_video(src, args.output, model=args.model)
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
                         camera_index=camera_index, save=args.save,
                         savedir=args.savedir, model=args.model)
    mapper.run()


if __name__ == "__main__":
    cli_main()
