#!/usr/bin/env python3
"""
3D World Mapper — Real-time monocular 3D point cloud reconstruction.

Uses webcam + monocular depth estimation + visual odometry to build
a colored 3D point cloud of the environment in real time.
"""

import os
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
BASE_DIR = Path(__file__).parent
SPLAT_DIR = BASE_DIR / "splat"
DEPTH_DIR = SPLAT_DIR / "depth_frames"
LOG_DIR = BASE_DIR / "logs"

for d in [SPLAT_DIR, DEPTH_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "mapper.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("mapper")


# ============================================================================
# 1. CAMERA CAPTURE
# ============================================================================
class CameraCapture:
    """OpenCV camera with auto-fallback and reconnection."""

    def __init__(self, width=640, height=480, fps=30):
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None
        self._open()

    def _open(self):
        for idx in (0, 1, 2):
            log.info(f"Trying camera index {idx}...")
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                cap.set(cv2.CAP_PROP_FPS, self.fps)
                self.cap = cap
                log.info(f"Camera opened on index {idx}")
                return
            cap.release()
        log.error("No camera found on indices 0-2")
        self.cap = None

    def read(self):
        """Returns (ok, frame). If camera lost, attempts reconnect."""
        if self.cap is None or not self.cap.isOpened():
            log.warning("Camera disconnected, attempting reconnect...")
            self._open()
            if self.cap is None:
                return False, None
        ok, frame = self.cap.read()
        if not ok:
            log.warning("Frame read failed, reconnecting...")
            self.cap.release()
            self.cap = None
            self._open()
            if self.cap is not None:
                ok, frame = self.cap.read()
        return ok, frame

    def release(self):
        if self.cap is not None:
            self.cap.release()


# ============================================================================
# 2. MONOCULAR DEPTH ESTIMATION
# ============================================================================
class DepthEstimator:
    """Depth Anything V2 (small) via transformers, with MiDaS fallback."""

    def __init__(self):
        self.pipe = None
        self.model_name = None
        self._load_model()

    def _load_model(self):
        from transformers import pipeline

        # Try Depth Anything V2 small first
        candidates = [
            "depth-anything/Depth-Anything-V2-Small-hf",
            "depth-anything/Depth-Anything-V2-Base-hf",
            "Intel/dpt-swinv2-tiny-256",     # DPT small
            "Intel/dpt-hybrid-midas",
        ]
        for name in candidates:
            try:
                log.info(f"Loading depth model: {name}")
                self.pipe = pipeline(
                    "depth-estimation",
                    model=name,
                    device="mps" if torch.backends.mps.is_available() else "cpu",
                )
                self.model_name = name
                log.info(f"Depth model loaded: {name}")
                return
            except Exception as e:
                log.warning(f"Failed to load {name}: {e}")

        raise RuntimeError("Could not load any depth estimation model")

    def estimate(self, frame_bgr):
        """Returns depth map as float32 HxW array (approximate meters)."""
        from PIL import Image as PILImage

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = PILImage.fromarray(rgb)

        result = self.pipe(pil_img)
        depth_raw = result["depth"]
        # Handle both PIL Image and numpy array returns
        if isinstance(depth_raw, PILImage.Image):
            depth = np.array(depth_raw, dtype=np.float32)
        else:
            depth = np.array(depth_raw, dtype=np.float32)

        # Resize to match input frame
        h, w = frame_bgr.shape[:2]
        if depth.shape != (h, w):
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)

        # Normalize: map to approximate meters (0.5 - 10m range)
        d_min, d_max = depth.min(), depth.max()
        if d_max - d_min > 1e-6:
            depth = (depth - d_min) / (d_max - d_min)
        depth = 0.5 + depth * 9.5  # [0.5, 10.0] meters

        return depth


# ============================================================================
# 3. CAMERA POSE ESTIMATION (Visual Odometry)
# ============================================================================
class PoseEstimator:
    """ORB-based frame-to-frame visual odometry."""

    def __init__(self, fx, fy, cx, cy):
        self.K = np.array([[fx, 0, cx],
                           [0, fy, cy],
                           [0,  0,  1]], dtype=np.float64)
        self.orb = cv2.ORB_create(nfeatures=1000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.prev_gray = None
        self.prev_kp = None
        self.prev_des = None
        self.pose = np.eye(4, dtype=np.float64)  # world pose (camera-to-world)
        self.tracking = True

    def update(self, frame_bgr):
        """Update pose from new frame. Returns (pose_4x4, keypoints, matches_img)."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        kp, des = self.orb.detectAndCompute(gray, None)

        matches_img = cv2.drawKeypoints(frame_bgr, kp, None,
                                        color=(0, 255, 0),
                                        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        if self.prev_des is None or des is None or len(kp) < 8:
            self.prev_gray = gray
            self.prev_kp = kp
            self.prev_des = des
            return self.pose.copy(), kp, matches_img

        # Match features
        raw_matches = self.bf.knnMatch(self.prev_des, des, k=2)

        # Lowe's ratio test
        good = []
        for m_pair in raw_matches:
            if len(m_pair) == 2:
                m, n = m_pair
                if m.distance < 0.75 * n.distance:
                    good.append(m)

        if len(good) < 15:
            log.warning(f"Tracking: only {len(good)} matches, holding pose")
            self.tracking = False
            self.prev_gray = gray
            self.prev_kp = kp
            self.prev_des = des
            return self.pose.copy(), kp, matches_img

        # Extract matched points
        pts1 = np.float64([self.prev_kp[m.queryIdx].pt for m in good])
        pts2 = np.float64([kp[m.trainIdx].pt for m in good])

        # Essential matrix
        E, mask = cv2.findEssentialMat(pts1, pts2, self.K,
                                       method=cv2.RANSAC,
                                       prob=0.999, threshold=1.0)
        if E is None:
            self.tracking = False
            self.prev_gray = gray
            self.prev_kp = kp
            self.prev_des = des
            return self.pose.copy(), kp, matches_img

        _, R, t, mask2 = cv2.recoverPose(E, pts1, pts2, self.K)

        # Build relative transform
        T_rel = np.eye(4, dtype=np.float64)
        T_rel[:3, :3] = R
        T_rel[:3, 3] = t.ravel()

        # Accumulate pose
        self.pose = self.pose @ np.linalg.inv(T_rel)
        self.tracking = True

        # Draw matches on image
        if self.prev_kp is not None:
            match_pairs = [(self.prev_kp[m.queryIdx].pt, kp[m.trainIdx].pt) for m in good[:50]]
            for (x1, y1), (x2, y2) in match_pairs:
                cv2.line(matches_img, (int(x2), int(y2)), (int(x1), int(y1)),
                         (0, 255, 255), 1)

        self.prev_gray = gray
        self.prev_kp = kp
        self.prev_des = des

        return self.pose.copy(), kp, matches_img


# ============================================================================
# 4. POINT CLOUD ACCUMULATION
# ============================================================================
class PointCloud:
    """Global point cloud with voxel downsampling."""

    def __init__(self, max_points=2_000_000, voxel_size=0.05):
        self.points = np.zeros((0, 3), dtype=np.float32)
        self.colors = np.zeros((0, 3), dtype=np.uint8)
        self.max_points = max_points
        self.voxel_size = voxel_size
        self.lock = Lock()
        self.frame_count = 0

    def add_frame(self, depth, frame_bgr, pose, K, subsample=4):
        """Back-project depth to 3D, transform to world coords, append."""
        h, w = depth.shape
        # Subsample grid
        ys = np.arange(0, h, subsample)
        xs = np.arange(0, w, subsample)
        xs, ys = np.meshgrid(xs, ys)
        xs = xs.ravel()
        ys = ys.ravel()

        z = depth[ys, xs]

        # Filter invalid depth
        valid = (z > 0.3) & (z < 15.0)
        xs, ys, z = xs[valid], ys[valid], z[valid]

        # Back-project to camera coordinates
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        x3d = (xs.astype(np.float32) - cx) * z / fx
        y3d = (ys.astype(np.float32) - cy) * z / fy
        z3d = z

        pts_cam = np.stack([x3d, y3d, z3d], axis=1)  # Nx3

        # Transform to world coordinates
        R = pose[:3, :3].astype(np.float32)
        t = pose[:3, 3].astype(np.float32)
        pts_world = (pts_cam @ R.T) + t

        # Get colors (BGR -> RGB)
        colors = frame_bgr[ys, xs][:, ::-1].copy()  # RGB uint8

        with self.lock:
            self.points = np.vstack([self.points, pts_world])
            self.colors = np.vstack([self.colors, colors])
            self.frame_count += 1

            # Voxel downsample every 30 frames
            if self.frame_count % 30 == 0:
                self._voxel_downsample()

    def _voxel_downsample(self):
        """Vectorized voxel grid downsampling using numpy."""
        if len(self.points) < 1000:
            return

        n_before = len(self.points)

        # Quantize to voxel grid
        voxel_indices = np.floor(self.points / self.voxel_size).astype(np.int64)

        # Pack 3 indices into a single int for fast unique lookup
        # Shift to positive range first
        voxel_indices -= voxel_indices.min(axis=0)
        dims = voxel_indices.max(axis=0) + 1
        flat = (voxel_indices[:, 0] * dims[1] * dims[2]
                + voxel_indices[:, 1] * dims[2]
                + voxel_indices[:, 2])

        _, keep = np.unique(flat, return_index=True)
        keep.sort()

        # If still too many, randomly subsample
        if len(keep) > self.max_points:
            keep = np.random.choice(keep, self.max_points, replace=False)
            keep.sort()

        self.points = self.points[keep]
        self.colors = self.colors[keep]

        log.info(f"Voxel downsample: {n_before} -> {len(self.points)} points")

    def get_data(self):
        with self.lock:
            return self.points.copy(), self.colors.copy()

    def count(self):
        return len(self.points)


# ============================================================================
# 5. PLY EXPORT (replaces Open3D dependency)
# ============================================================================
def save_ply(filepath, points, colors):
    """Save point cloud as PLY file."""
    n = len(points)
    if n == 0:
        log.warning("No points to save")
        return

    filepath = str(filepath)
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

    # Build binary data
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

    with open(filepath, 'wb') as f:
        f.write(header.encode('ascii'))
        f.write(data.tobytes())

    log.info(f"Saved {n} points to {filepath}")


# ============================================================================
# 6. 3D VISUALIZER (PyVista)
# ============================================================================
class Visualizer3D:
    """Non-blocking 3D point cloud viewer using PyVista."""

    def __init__(self):
        self.plotter = None
        self.actor = None
        self._initialized = False

    def init(self):
        """Initialize the PyVista plotter (must be called from main thread)."""
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
        """Update displayed point cloud."""
        if not self._initialized or len(points) < 10:
            return

        try:
            import pyvista as pv

            cloud = pv.PolyData(points.astype(np.float32))
            # Colors need to be 0-255 uint8, shape (N, 3)
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
    """Main application orchestrating all modules."""

    def __init__(self):
        self.running = True
        self.camera = None
        self.depth_estimator = None
        self.pose_estimator = None
        self.cloud = PointCloud()
        self.viz3d = Visualizer3D()

        # Timing
        self.last_save = time.time()
        self.last_backup = time.time()
        self.last_status = time.time()
        self.last_depth_save = 0
        self.frame_idx = 0
        self.fps_history = []

        # Signal handler
        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, sig, frame):
        log.info("SIGINT received, shutting down gracefully...")
        self.running = False

    def _save_state(self, tag=""):
        """Save current point cloud."""
        pts, cols = self.cloud.get_data()
        if len(pts) == 0:
            return

        # Latest
        save_ply(SPLAT_DIR / "cloud_latest.ply", pts, cols)

        if tag:
            save_ply(SPLAT_DIR / f"cloud_{tag}.ply", pts, cols)

    def run(self):
        log.info("=" * 60)
        log.info("3D World Mapper starting")
        log.info("=" * 60)

        # ── Init camera ────────────────────────────────────────────
        log.info("Initializing camera...")
        self.camera = CameraCapture()
        if self.camera.cap is None:
            log.error("No camera available. Exiting.")
            return

        # Read one frame to get actual resolution
        ok, test_frame = self.camera.read()
        if not ok:
            log.error("Cannot read from camera. Exiting.")
            return

        h, w = test_frame.shape[:2]
        log.info(f"Camera resolution: {w}x{h}")

        # ── Estimate camera intrinsics ─────────────────────────────
        fx = fy = w * 0.8  # Rough approximation
        cx, cy = w / 2.0, h / 2.0
        K = np.array([[fx, 0, cx],
                      [0, fy, cy],
                      [0,  0,  1]], dtype=np.float64)
        log.info(f"Camera intrinsics: fx={fx:.0f} fy={fy:.0f} cx={cx:.0f} cy={cy:.0f}")

        # ── Init depth estimator ───────────────────────────────────
        log.info("Loading depth estimation model...")
        try:
            self.depth_estimator = DepthEstimator()
        except Exception as e:
            log.error(f"Failed to load depth model: {e}")
            return

        # ── Init pose estimator ────────────────────────────────────
        self.pose_estimator = PoseEstimator(fx, fy, cx, cy)

        # ── Init 3D visualizer ─────────────────────────────────────
        self.viz3d.init()

        # ── Create OpenCV windows ──────────────────────────────────
        cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Depth", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Features", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Camera", 480, 360)
        cv2.resizeWindow("Depth", 480, 360)
        cv2.resizeWindow("Features", 480, 360)
        # Arrange side by side
        cv2.moveWindow("Camera", 0, 0)
        cv2.moveWindow("Depth", 490, 0)
        cv2.moveWindow("Features", 980, 0)

        log.info("Starting main loop")
        log.info("Press 'q' in any OpenCV window or Ctrl+C to quit")

        # ── Main loop ──────────────────────────────────────────────
        try:
            while self.running:
                t_start = time.time()

                # Read frame
                ok, frame = self.camera.read()
                if not ok:
                    log.warning("No frame, waiting for camera...")
                    self._save_state("disconnect")
                    time.sleep(1.0)
                    continue

                self.frame_idx += 1

                # ── Depth estimation ───────────────────────────────
                depth = None
                try:
                    depth = self.depth_estimator.estimate(frame)
                except Exception as e:
                    log.warning(f"Depth estimation failed: {e}")

                # ── Pose estimation ────────────────────────────────
                pose = None
                matches_img = frame.copy()
                try:
                    pose, kp, matches_img = self.pose_estimator.update(frame)
                except Exception as e:
                    log.warning(f"Pose estimation failed: {e}")

                # ── Point cloud accumulation ───────────────────────
                if depth is not None and pose is not None:
                    try:
                        self.cloud.add_frame(depth, frame, pose, K)
                    except Exception as e:
                        log.warning(f"Point cloud accumulation failed: {e}")

                # ── Display ────────────────────────────────────────
                dt = time.time() - t_start
                fps = 1.0 / dt if dt > 0 else 0
                self.fps_history.append(fps)
                if len(self.fps_history) > 30:
                    self.fps_history = self.fps_history[-30:]
                avg_fps = sum(self.fps_history) / len(self.fps_history)

                # Camera feed with FPS
                disp_frame = frame.copy()
                cv2.putText(disp_frame, f"FPS: {avg_fps:.1f}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 255, 0), 2)
                cv2.putText(disp_frame, f"Points: {self.cloud.count():,}",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 0), 2)
                cv2.imshow("Camera", disp_frame)

                # Depth map
                if depth is not None:
                    depth_norm = ((depth - depth.min()) / (depth.max() - depth.min() + 1e-6) * 255).astype(np.uint8)
                    depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)
                    cv2.imshow("Depth", depth_color)

                # Features
                cv2.imshow("Features", matches_img)

                # ── 3D Visualization ───────────────────────────────
                if self.frame_idx % 5 == 0:  # Update 3D view every 5 frames
                    pts, cols = self.cloud.get_data()
                    self.viz3d.update(pts, cols)

                # ── Save depth frames ──────────────────────────────
                if depth is not None and self.frame_idx % 5 == 0:
                    try:
                        depth_16 = (depth * 1000).astype(np.uint16)  # mm
                        fname = DEPTH_DIR / f"depth_{self.frame_idx:06d}.png"
                        cv2.imwrite(str(fname), depth_16)
                    except Exception as e:
                        log.debug(f"Depth save failed: {e}")

                # ── Periodic saves ─────────────────────────────────
                now = time.time()
                if now - self.last_save >= 10:
                    self._save_state()
                    self.last_save = now

                if now - self.last_backup >= 60:
                    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    self._save_state(ts)
                    self.last_backup = now

                # ── Status line ────────────────────────────────────
                if now - self.last_status >= 5:
                    tracking = "OK" if self.pose_estimator.tracking else "LOST"
                    last_save_ago = now - self.last_save
                    log.info(
                        f"STATUS | FPS: {avg_fps:.1f} | Points: {self.cloud.count():,} | "
                        f"Tracking: {tracking} | Last save: {last_save_ago:.0f}s ago"
                    )
                    self.last_status = now

                # ── Check for quit ─────────────────────────────────
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    log.info("'q' pressed, shutting down...")
                    self.running = False

        except Exception as e:
            log.error(f"Main loop error: {e}", exc_info=True)
        finally:
            self._shutdown()

    def _shutdown(self):
        log.info("Shutting down...")

        # Final save
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self._save_state(f"final_{ts}")

        # Cleanup
        if self.camera:
            self.camera.release()
        cv2.destroyAllWindows()
        self.viz3d.close()

        log.info(f"Final point count: {self.cloud.count():,}")
        log.info("Shutdown complete")


# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    mapper = WorldMapper()
    mapper.run()
