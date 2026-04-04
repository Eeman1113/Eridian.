#!/usr/bin/env python3
"""
SLAM Module for Eridian
Handles camera pose tracking using visual odometry.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List
import threading

from .logging import get_logger, get_monitor, safe_execute, timer_context


class PoseTracker:
    """
    Camera pose tracker using Lucas-Kanade optical flow and essential matrix estimation.
    Provides 6-DOF camera pose estimation from monocular video.
    """
    
    def __init__(self, K: np.ndarray,
                 max_features: int = 500,
                 feature_quality: float = 0.03,
                 feature_min_distance: int = 7,
                 feature_refresh_interval: int = 20,
                 ransac_threshold: float = 1.0):
        """
        Initialize pose tracker.
        
        Args:
            K: Camera intrinsic matrix (3x3)
            max_features: Maximum number of features to track
            feature_quality: Quality level for feature detection (0.01-1.0)
            feature_min_distance: Minimum distance between features in pixels
            feature_refresh_interval: Refresh features every N frames
            ransac_threshold: RANSAC threshold for essential matrix (pixels)
        """
        self.logger = get_logger()
        self.monitor = get_monitor()
        self.K = K.copy()
        
        # Optical flow parameters
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
        
        # Feature detection parameters
        self.gft_params = dict(
            maxCorners=max_features,
            qualityLevel=feature_quality,
            minDistance=feature_min_distance,
            blockSize=7
        )
        
        self.feature_refresh_interval = feature_refresh_interval
        self.ransac_threshold = ransac_threshold
        
        # Pose state
        self.R = np.eye(3, dtype=np.float64)  # Rotation
        self.t = np.zeros((3, 1), dtype=np.float64)  # Translation
        self.scale = 1.0  # Scale factor for monocular VO
        
        # Tracking state
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_pts: Optional[np.ndarray] = None
        self.frame_count = 0
        self._lock = threading.Lock()
        
        self.logger.info("Pose tracker initialized")
    
    @safe_execute("pose_update", default_return=np.eye(4), critical=False, context="pose tracking")
    def update(self, gray: np.ndarray, depth: Optional[np.ndarray] = None,
               max_depth: float = 8.0) -> np.ndarray:
        """
        Update camera pose from new frame.
        
        Args:
            gray: Grayscale image
            depth: Optional depth map for scale estimation
            max_depth: Maximum depth for scale estimation (meters)
            
        Returns:
            Current camera pose as 4x4 transformation matrix
        """
        self.frame_count += 1
        
        with self._lock:
            # Initialize on first frame
            if self.prev_gray is None:
                self._detect_features(gray)
                self.prev_gray = gray
                return self._get_pose()
            
            # Not enough features detected
            if self.prev_pts is None or len(self.prev_pts) < 10:
                self._detect_features(gray)
                self.prev_gray = gray
                return self._get_pose()
            
            # Track features with optical flow
            with timer_context(self.logger, "Optical flow", log_threshold=0.02):
                next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                    self.prev_gray, gray,
                    self.prev_pts, None,
                    **self.lk_params
                )
            
            if next_pts is None or status is None:
                self._detect_features(gray)
                self.prev_gray = gray
                return self._get_pose()
            
            # Filter good tracks
            ok = status.ravel() == 1
            if ok.sum() < 8:
                self._detect_features(gray)
                self.prev_gray = gray
                return self._get_pose()
            
            prev_good = self.prev_pts[ok]
            next_good = next_pts[ok]
            
            # Estimate essential matrix
            try:
                with timer_context(self.logger, "Essential matrix", log_threshold=0.01):
                    E, mask = cv2.findEssentialMat(
                        prev_good, next_good, self.K,
                        method=cv2.RANSAC,
                        prob=0.999,
                        threshold=self.ransac_threshold
                    )
                
                if E is None:
                    raise ValueError("No essential matrix found")
                
                # Recover pose
                _, R_rel, t_rel, inlier_mask = cv2.recoverPose(
                    E, prev_good, next_good, self.K
                )
                
                # Estimate scale from depth if available
                if depth is not None and inlier_mask is not None:
                    scale = self._estimate_scale(
                        prev_good[inlier_mask.ravel() > 0],
                        depth,
                        max_depth
                    )
                    # Smooth scale updates
                    self.scale = 0.85 * self.scale + 0.15 * scale
                
                # Update pose
                self.R = self.R @ R_rel.T
                self.t = self.t - self.scale * self.R @ t_rel
                
                self.monitor.increment_counter("pose_updates")
                
            except Exception as e:
                self.logger.warning(f"Pose estimation failed: {e}")
                self._detect_features(gray)
            
            # Refresh features periodically or when tracking degrades
            if ok.sum() >= 50 and self.frame_count % self.feature_refresh_interval != 0:
                self.prev_pts = next_good.reshape(-1, 1, 2)
            else:
                self._detect_features(gray)
            
            self.prev_gray = gray
        
        return self._get_pose()
    
    def _detect_features(self, gray: np.ndarray):
        """Detect good features to track."""
        self.prev_pts = cv2.goodFeaturesToTrack(gray, mask=None, **self.gft_params)
        self.monitor.increment_counter("feature_detections")
    
    def _estimate_scale(self, points: np.ndarray, depth: np.ndarray, 
                       max_depth: float) -> float:
        """
        Estimate scale factor from depth at tracked points.
        
        Args:
            points: Feature points in image coordinates (N, 2)
            depth: Depth map
            max_depth: Maximum depth in meters
            
        Returns:
            Estimated scale factor
        """
        h, w = depth.shape
        depths = []
        
        # Sample depth at feature points
        for pt in points[:60]:  # Limit to 60 points for speed
            x, y = int(pt[0]), int(pt[1])
            x = np.clip(x, 0, w - 1)
            y = np.clip(y, 0, h - 1)
            
            d = depth[y, x]
            if 0.05 < d < 0.95:  # Filter outliers
                # Convert normalized depth to metric depth
                metric_depth = 0.1 + (1.0 - d) * (max_depth - 0.1)
                depths.append(metric_depth)
        
        if len(depths) > 3:
            median_depth = np.median(depths)
            # Scale factor: translation magnitude relative to scene scale
            scale = median_depth * 0.08  # Heuristic scaling
            return np.clip(scale, 0.01, 1.0)
        
        return 1.0  # Default scale if estimation fails
    
    def _get_pose(self) -> np.ndarray:
        """Get current camera pose as 4x4 transformation matrix."""
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = self.R
        T[:3, 3] = self.t.ravel()
        return T
    
    def get_position(self) -> np.ndarray:
        """Get current camera position in world coordinates."""
        return self.t.ravel()
    
    def get_orientation(self) -> np.ndarray:
        """Get current camera orientation matrix."""
        return self.R.copy()
    
    def reset(self):
        """Reset pose to origin."""
        with self._lock:
            self.R = np.eye(3, dtype=np.float64)
            self.t = np.zeros((3, 1), dtype=np.float64)
            self.scale = 1.0
            self.prev_gray = None
            self.prev_pts = None
            self.frame_count = 0
            self.logger.info("Pose tracker reset")
    
    def get_trajectory(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Get trajectory history (not implemented in current version).
        
        Returns:
            Tuple of (positions, orientations) lists
        """
        self.logger.warning("Trajectory history not implemented")
        return [], []


class SLAMPipeline:
    """
    Complete SLAM pipeline integrating pose tracking with depth estimation.
    """
    
    def __init__(self, K: np.ndarray, depth_estimator, config: dict = None):
        """
        Initialize SLAM pipeline.
        
        Args:
            K: Camera intrinsic matrix
            depth_estimator: DepthEstimator instance
            config: Configuration dictionary
        """
        self.logger = get_logger()
        self.monitor = get_monitor()
        
        config = config or {}
        
        self.pose_tracker = PoseTracker(
            K=K,
            max_features=config.get('max_features', 500),
            feature_quality=config.get('feature_quality', 0.03),
            feature_min_distance=config.get('feature_min_distance', 7),
            feature_refresh_interval=config.get('feature_refresh_interval', 20),
            ransac_threshold=config.get('ransac_threshold', 1.0)
        )
        
        self.depth_estimator = depth_estimator
        self.max_depth = config.get('max_depth', 8.0)
        
        # Trajectory history
        self.trajectory: List[np.ndarray] = []
    
    def process_frame(self, gray: np.ndarray, rgb: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Process a frame through the SLAM pipeline.
        
        Args:
            gray: Grayscale image
            rgb: RGB image for depth estimation
            
        Returns:
            Tuple of (pose, depth)
        """
        # Estimate depth
        depth = self.depth_estimator.estimate(rgb)
        
        # Update pose
        pose = self.pose_tracker.update(gray, depth, self.max_depth)
        
        # Record trajectory
        self.trajectory.append(pose[:3, 3].copy())
        
        return pose, depth
    
    def get_trajectory(self) -> np.ndarray:
        """Get camera trajectory as array of positions."""
        if not self.trajectory:
            return np.zeros((0, 3))
        return np.array(self.trajectory)
    
    def reset(self):
        """Reset the SLAM pipeline."""
        self.pose_tracker.reset()
        self.trajectory.clear()
        self.logger.info("SLAM pipeline reset")


if __name__ == "__main__":
    # Test the SLAM module
    from .logging import setup_logging
    import cv2
    
    logger = setup_logging(level="INFO")
    
    # Create dummy camera intrinsics
    K = np.array([
        [500, 0, 320],
        [0, 500, 240],
        [0, 0, 1]
    ], dtype=np.float64)
    
    # Create pose tracker
    tracker = PoseTracker(K)
    
    # Test with camera input
    try:
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            pose = tracker.update(gray)
            
            print(f"Initial pose:\n{pose}")
            print(f"Position: {tracker.get_position()}")
            print(f"Frame count: {tracker.frame_count}")
            
            # Process a few more frames
            for i in range(5):
                ret, frame = cap.read()
                if ret:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    pose = tracker.update(gray)
                    print(f"Frame {i+1}: Position {tracker.get_position()}")
        
        cap.release()
        
    except Exception as e:
        print(f"Test failed: {e}")
    
    print("SLAM module test completed ✓")