#!/usr/bin/env python3
"""
Depth Estimation Module for Eridian
Handles monocular depth estimation using neural networks.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Union
import threading

from .logging import get_logger, get_monitor, safe_execute, timer_context


class DepthEstimator:
    """
    Monocular depth estimator using MiDaS models.
    Supports CPU, CUDA, and MPS (Apple Silicon) backends.
    """
    
    def __init__(self, model_name: str = "MiDaS_small", 
                 max_depth: float = 8.0,
                 min_depth: float = 0.1,
                 device: Optional[str] = None):
        """
        Initialize depth estimator.
        
        Args:
            model_name: Name of MiDaS model ("MiDaS_small" or "DPT_Hybrid")
            max_depth: Maximum depth in meters
            min_depth: Minimum depth in meters
            device: Specific device to use ("cuda", "mps", "cpu"), or None for auto-detect
        """
        self.logger = get_logger()
        self.monitor = get_monitor()
        self.model_name = model_name
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.device = self._select_device(device)
        self.model = None
        self.transform = None
        self._lock = threading.Lock()
        
        self._load_model()
    
    def _select_device(self, device: Optional[str]) -> torch.device:
        """Select appropriate device for inference."""
        if device is not None:
            return torch.device(device)
        
        if torch.cuda.is_available():
            self.logger.info("CUDA device available")
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.logger.info("MPS (Apple Silicon) device available")
            return torch.device("mps")
        else:
            self.logger.info("Using CPU device")
            return torch.device("cpu")
    
    def _load_model(self):
        """Load the MiDaS model and transforms."""
        try:
            self.logger.info(f"Loading {self.model_name} model on {self.device}...")
            
            with timer_context(self.logger, "Model loading", log_threshold=0.5):
                # Load model
                self.model = torch.hub.load(
                    "intel-isl/MiDaS",
                    self.model_name,
                    trust_repo=True
                )
                self.model.to(self.device).eval()
                
                # Load transforms
                transforms = torch.hub.load(
                    "intel-isl/MiDaS",
                    "transforms",
                    trust_repo=True
                )
                
                if "small" in self.model_name.lower():
                    self.transform = transforms.small_transform
                else:
                    self.transform = transforms.dpt_transform
            
            self.logger.info(f"✓ {self.model_name} loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    @torch.inference_mode()
    @safe_execute("depth_estimation", default_return=None, critical=False, context="depth prediction")
    def estimate(self, rgb: np.ndarray) -> Optional[np.ndarray]:
        """
        Estimate depth from RGB image.
        
        Args:
            rgb: RGB image as numpy array (H, W, 3), values in [0, 255]
            
        Returns:
            Normalized depth map (H, W) in [0, 1] range, or None if estimation failed
        """
        if self.model is None or self.transform is None:
            self.logger.error("Model not loaded")
            return None
        
        if rgb is None or rgb.size == 0:
            self.logger.warning("Invalid input image")
            return None
        
        with self._lock:
            try:
                with timer_context(self.logger, "Depth inference", log_threshold=0.05):
                    # Prepare input
                    input_tensor = self.transform(rgb).to(self.device)
                    
                    # Run inference
                    prediction = self.model(input_tensor)
                    
                    # Resize to original image size
                    prediction = F.interpolate(
                        prediction.unsqueeze(1),
                        size=rgb.shape[:2],
                        mode="bicubic",
                        align_corners=False
                    ).squeeze()
                    
                    # Convert to numpy
                    depth = prediction.cpu().numpy()
                    
                    # Normalize to [0, 1]
                    lo, hi = depth.min(), depth.max()
                    if hi > lo:
                        depth = (depth - lo) / (hi - lo)
                    else:
                        depth = np.zeros_like(depth)
                    
                    self.monitor.increment_counter("depth_estimates")
                    
                    return depth.astype(np.float32)
                    
            except Exception as e:
                self.logger.error(f"Depth estimation failed: {e}")
                return None
    
    def estimate_metric(self, rgb: np.ndarray) -> Optional[np.ndarray]:
        """
        Estimate metric depth in meters.
        
        Args:
            rgb: RGB image as numpy array (H, W, 3)
            
        Returns:
            Metric depth map (H, W) in meters, or None if estimation failed
        """
        depth_normalized = self.estimate(rgb)
        if depth_normalized is None:
            return None
        
        # Convert normalized depth to metric depth
        # Note: This is an approximation. MiDaS outputs inverse depth (disparity).
        # For accurate metric depth, additional calibration may be needed.
        metric_depth = self.min_depth + (1.0 - depth_normalized) * (self.max_depth - self.min_depth)
        
        return metric_depth
    
    def get_disparity(self, rgb: np.ndarray) -> Optional[np.ndarray]:
        """
        Get disparity map (inverse depth).
        
        Args:
            rgb: RGB image as numpy array (H, W, 3)
            
        Returns:
            Disparity map (H, W), higher values = closer objects
        """
        depth_normalized = self.estimate(rgb)
        if depth_normalized is None:
            return None
        
        # Depth is already normalized such that 1 = close, 0 = far (disparity)
        return depth_normalized
    
    def get_model_info(self) -> dict:
        """Get information about the depth estimator."""
        return {
            "model_name": self.model_name,
            "device": str(self.device),
            "max_depth": self.max_depth,
            "min_depth": self.min_depth,
            "is_loaded": self.model is not None,
        }
    
    def change_model(self, new_model_name: str):
        """
        Change the depth estimation model.
        
        Args:
            new_model_name: Name of new model ("MiDaS_small" or "DPT_Hybrid")
        """
        if new_model_name == self.model_name:
            self.logger.info(f"Already using {new_model_name}")
            return
        
        self.logger.info(f"Changing model from {self.model_name} to {new_model_name}")
        self.model_name = new_model_name
        self._load_model()


class DepthEstimatorFactory:
    """Factory for creating depth estimators with predefined configurations."""
    
    @staticmethod
    def create_fast() -> DepthEstimator:
        """Create fast depth estimator (MiDaS_small, CPU/MPS)."""
        return DepthEstimator(
            model_name="MiDaS_small",
            max_depth=8.0,
            min_depth=0.1,
        )
    
    @staticmethod
    def create_accurate() -> DepthEstimator:
        """Create accurate depth estimator (DPT_Hybrid, GPU if available)."""
        return DepthEstimator(
            model_name="DPT_Hybrid",
            max_depth=10.0,
            min_depth=0.1,
        )
    
    @staticmethod
    def create_balanced() -> DepthEstimator:
        """Create balanced depth estimator (MiDaS_small, GPU if available)."""
        return DepthEstimator(
            model_name="MiDaS_small",
            max_depth=8.0,
            min_depth=0.1,
        )


if __name__ == "__main__":
    # Test the depth estimation module
    from .logging import setup_logging
    import cv2
    
    logger = setup_logging(level="INFO")
    
    # Create depth estimator
    estimator = DepthEstimatorFactory.create_fast()
    print(f"Model info: {estimator.get_model_info()}")
    
    # Test with a sample image (if camera available)
    try:
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            depth = estimator.estimate(rgb)
            
            if depth is not None:
                print(f"Depth estimation successful: {depth.shape}")
                print(f"Depth range: [{depth.min():.3f}, {depth.max():.3f}]")
            else:
                print("Depth estimation failed")
        else:
            print("Could not capture test frame")
            
    except Exception as e:
        print(f"Test failed: {e}")
    
    print("Depth estimation module test completed ✓")