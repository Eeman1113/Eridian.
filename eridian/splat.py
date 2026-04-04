#!/usr/bin/env python3
"""
Gaussian Splat Module for Eridian
Handles building, managing, and saving 3D Gaussian splats.
"""

import numpy as np
import os
from pathlib import Path
from typing import Optional, Tuple
import threading
import time

from .logging import get_logger, get_monitor, safe_execute, timer_context


class SplatBuilder:
    """
    Builder for 3D Gaussian splats from depth+RGB+pose data.
    Supports incremental updates, downsampling, and atomic saves.
    """
    
    def __init__(self, max_points: int = 800000, 
                 point_step: int = 4,
                 output_dir: str = "./splat",
                 downsample_ratio: float = 0.8,
                 downsample_interval: int = 8):
        """
        Initialize splat builder.
        
        Args:
            max_points: Maximum number of points to maintain
            point_step: Sample every Nth pixel (higher = sparser)
            output_dir: Directory for saving splat files
            downsample_ratio: Target ratio when downsampling (0-1)
            downsample_interval: Check downsampling every N frames
        """
        self.logger = get_logger()
        self.monitor = get_monitor()
        
        self.max_points = max_points
        self.point_step = point_step
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.downsample_ratio = downsample_ratio
        self.downsample_interval = downsample_interval
        
        # Splat data (thread-safe)
        self._lock = threading.Lock()
        self._positions: list[np.ndarray] = []
        self._colors: list[np.ndarray] = []
        self._scales: list[np.ndarray] = []
        self._opacities: list[np.ndarray] = []
        self._rotations: list[np.ndarray] = []
        
        # Statistics
        self.total_points = 0
        self.frame_count = 0
        self._save_count = 0
        
        self.logger.info(f"Splat builder initialized (max_points={max_points}, step={point_step})")
    
    @safe_execute("splat_add", default_return=None, critical=False, context="adding points to splat")
    def add(self, rgb: np.ndarray, depth: np.ndarray, pose: np.ndarray, 
            K: np.ndarray, point_step: Optional[int] = None,
            min_depth: float = 0.05, max_depth: float = 0.92,
            max_metric_depth: float = 8.0):
        """
        Add points from a frame to the splat.
        
        Args:
            rgb: RGB image (H, W, 3)
            depth: Normalized depth map (H, W)
            pose: Camera pose as 4x4 transformation matrix
            K: Camera intrinsic matrix (3x3)
            point_step: Override default point step
            min_depth: Minimum normalized depth threshold
            max_depth: Maximum normalized depth threshold
            max_metric_depth: Maximum metric depth in meters
        """
        step = point_step if point_step is not None else self.point_step
        
        h, w = depth.shape
        
        # Sample pixels
        us = np.arange(0, w, step)
        vs = np.arange(0, h, step)
        uu, vv = np.meshgrid(us, vs)
        uu, vv = uu.ravel(), vv.ravel()
        
        # Filter by depth
        d = depth[vv, uu]
        ok = (d > min_depth) & (d < max_depth)
        
        if ok.sum() < 5:
            self.logger.debug(f"Not enough valid points ({ok.sum()})")
            return
        
        d, uu, vv = d[ok], uu[ok], vv[ok]
        
        # Convert normalized depth to metric depth
        dm = 0.1 + (1.0 - d) * (max_metric_depth - 0.1)
        
        # Back-project to 3D camera coordinates
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        pts_cam = np.stack([
            (uu - cx) * dm / fx,
            (vv - cy) * dm / fy,
            dm,
            np.ones_like(dm)
        ], axis=1)
        
        # Transform to world coordinates
        pts_world = (pose @ pts_cam.T).T[:, :3].astype(np.float32)
        
        # Extract colors
        colors = rgb[vv, uu, :].astype(np.float32) / 255.0
        
        # Compute scales based on depth and pixel footprint
        sv = np.clip(dm / fx * step * 0.5, 0.002, 0.15).astype(np.float32)
        scales = np.stack([sv, sv, sv * 0.25], axis=1)
        
        # Compute opacity based on depth confidence
        opacity = np.clip(0.9 - d * 0.3, 0.2, 0.95).astype(np.float32)
        
        # Identity rotation (quaternion w=1)
        rotations = np.zeros((len(pts_world), 4), np.float32)
        rotations[:, 0] = 1.0
        
        # Add to splat
        with self._lock:
            self._positions.append(pts_world)
            self._colors.append(colors)
            self._scales.append(scales)
            self._opacities.append(opacity)
            self._rotations.append(rotations)
            
            self.total_points += len(pts_world)
            self.frame_count += 1
            
            self.monitor.increment_counter("points_added", len(pts_world))
        
        # Check if we need to downsample
        if self.total_points > self.max_points and self.frame_count % self.downsample_interval == 0:
            self._downsample()
    
    def _downsample(self):
        """Downsample the splat to reduce memory usage."""
        with self._lock:
            with timer_context(self.logger, "Downsampling", log_threshold=0.1):
                # Concatenate all batches
                pos = np.concatenate(self._positions)
                col = np.concatenate(self._colors)
                sca = np.concatenate(self._scales)
                opa = np.concatenate(self._opacities)
                rot = np.concatenate(self._rotations)
                
                target = int(self.max_points * self.downsample_ratio)
                
                if len(pos) > target:
                    # Random sampling for uniform downsampling
                    idx = np.random.choice(len(pos), target, replace=False)
                    pos, col, sca, opa, rot = (
                        pos[idx], col[idx], sca[idx], opa[idx], rot[idx]
                    )
                
                # Store as single batches
                self._positions = [pos]
                self._colors = [col]
                self._scales = [sca]
                self._opacities = [opa]
                self._rotations = [rot]
                self.total_points = len(pos)
                
                self.logger.info(f"Downsampled to {self.total_points:,} points")
    
    def get_arrays(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Get position and color arrays for visualization.
        
        Returns:
            Tuple of (positions, colors) or (None, None) if empty
        """
        with self._lock:
            if not self._positions:
                return None, None
            return np.concatenate(self._positions), np.concatenate(self._colors)
    
    def get_full_arrays(self) -> Optional[dict]:
        """
        Get all splat arrays.
        
        Returns:
            Dictionary with 'positions', 'colors', 'scales', 'opacities', 'rotations'
            or None if empty
        """
        with self._lock:
            if not self._positions:
                return None
            
            return {
                'positions': np.concatenate(self._positions),
                'colors': np.concatenate(self._colors),
                'scales': np.concatenate(self._scales),
                'opacities': np.concatenate(self._opacities),
                'rotations': np.concatenate(self._rotations),
            }
    
    @safe_execute("splat_save", default_return=False, critical=False, context="saving splat")
    def save(self, filename: str = "splat.ply") -> bool:
        """
        Save splat to a standard 3DGS .ply file (atomic write).
        
        Args:
            filename: Output filename
            
        Returns:
            True if save successful, False otherwise
        """
        path = self.output_dir / filename
        arrays = self.get_full_arrays()
        
        if arrays is None:
            self.logger.warning("Nothing to save")
            return False
        
        with timer_context(self.logger, "Saving PLY", log_threshold=0.1):
            pos = arrays['positions']
            col = arrays['colors']
            sca = arrays['scales']
            opa = arrays['opacities']
            rot = arrays['rotations']
            
            n = len(pos)
            C0 = 0.28209479177387814  # SH zero-order coefficient
            
            # Convert to 3DGS format
            sh = (col - 0.5) / C0  # Spherical harmonics coefficients
            sl = np.log(np.clip(sca, 1e-7, None))  # Log scales
            ol = np.log(np.clip(opa, 1e-6, 1 - 1e-6) / 
                       (1 - np.clip(opa, 1e-6, 1 - 1e-6)))  # Logit opacity
            
            # Prepare PLY data
            header = (
                f"ply\nformat binary_little_endian 1.0\nelement vertex {n}\n"
                "property float x\nproperty float y\nproperty float z\n"
                "property float nx\nproperty float ny\nproperty float nz\n"
                "property float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\n"
                "property float opacity\n"
                "property float scale_0\nproperty float scale_1\nproperty float scale_2\n"
                "property float rot_0\nproperty float rot_1\nproperty float rot_2\nproperty float rot_3\n"
                "end_header\n"
            )
            
            data = np.zeros((n, 17), np.float32)
            data[:, 0:3] = pos    # x, y, z
            data[:, 6:9] = sh      # f_dc_0, f_dc_1, f_dc_2
            data[:, 9] = ol        # opacity
            data[:, 10:13] = sl    # scale_0, scale_1, scale_2
            data[:, 13:17] = rot   # rot_0, rot_1, rot_2, rot_3
            
            # Atomic write: write to temp file, then rename
            tmp_path = str(path) + ".tmp"
            
            try:
                with open(tmp_path, "wb") as f:
                    f.write(header.encode())
                    f.write(data.tobytes())
                
                os.replace(tmp_path, str(path))
                
                self._save_count += 1
                self.monitor.increment_counter("splat_saves")
                
                self.logger.info(f"✓ Saved {n:,} gaussians to {path}")
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to save splat: {e}")
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                return False
    
    def reset(self):
        """Reset the splat builder."""
        with self._lock:
            self._positions.clear()
            self._colors.clear()
            self._scales.clear()
            self._opacities.clear()
            self._rotations.clear()
            self.total_points = 0
            self.frame_count = 0
            self.logger.info("Splat builder reset")
    
    def get_stats(self) -> dict:
        """Get statistics about the splat."""
        with self._lock:
            return {
                'total_points': self.total_points,
                'frame_count': self.frame_count,
                'save_count': self._save_count,
                'batches': len(self._positions),
            }
    
    def estimate_memory(self) -> float:
        """
        Estimate memory usage in MB.
        
        Returns:
            Estimated memory usage in megabytes
        """
        stats = self.get_stats()
        # Each point: position(12) + color(12) + scale(12) + opacity(4) + rotation(16) = 56 bytes
        bytes_per_point = 56
        total_bytes = stats['total_points'] * bytes_per_point
        return total_bytes / (1024 * 1024)


class SplatManager:
    """
    Manager for multiple splat files and operations.
    """
    
    def __init__(self, output_dir: str = "./splat"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger()
    
    def list_splats(self) -> list[Path]:
        """List all .ply files in the output directory."""
        return sorted(self.output_dir.glob("*.ply"))
    
    def get_latest_splat(self) -> Optional[Path]:
        """Get the most recently modified .ply file."""
        splats = self.list_splats()
        if not splats:
            return None
        return max(splats, key=lambda p: p.stat().st_mtime)
    
    def cleanup_old_splats(self, keep: int = 5):
        """
        Remove old splat files, keeping only the most recent ones.
        
        Args:
            keep: Number of most recent files to keep
        """
        splats = self.list_splats()
        if len(splats) > keep:
            to_remove = splats[:-keep]
            for splat in to_remove:
                try:
                    splat.unlink()
                    self.logger.info(f"Removed old splat: {splat.name}")
                except Exception as e:
                    self.logger.warning(f"Failed to remove {splat.name}: {e}")


if __name__ == "__main__":
    # Test the splat module
    from .logging import setup_logging
    
    logger = setup_logging(level="INFO")
    
    # Create splat builder
    builder = SplatBuilder(max_points=100000, point_step=4)
    
    # Create dummy data
    rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    depth = np.random.rand(480, 640).astype(np.float32)
    pose = np.eye(4, dtype=np.float64)
    K = np.array([
        [500, 0, 320],
        [0, 500, 240],
        [0, 0, 1]
    ], dtype=np.float64)
    
    # Add some frames
    for i in range(3):
        builder.add(rgb, depth, pose, K)
        print(f"Frame {i+1}: {builder.get_stats()}")
    
    # Save
    builder.save("test_splat.ply")
    
    # Test manager
    manager = SplatManager()
    print(f"Splats: {[s.name for s in manager.list_splats()]}")
    print(f"Latest: {manager.get_latest_splat()}")
    
    print("Splat module test completed ✓")