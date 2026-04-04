#!/usr/bin/env python3
"""
Camera Module for Eridian
Handles camera input, device selection, and frame capture.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List
import threading

from .logging import get_logger, get_monitor, safe_execute, timer_context


class CameraDevice:
    """Represents a camera device with metadata."""
    
    def __init__(self, index: int, width: int, height: int, name: str = ""):
        self.index = index
        self.width = width
        self.height = height
        self.name = name or f"Camera {index}"
        self.resolution = width * height
    
    def __repr__(self) -> str:
        return f"CameraDevice(index={self.index}, {self.width}x{self.height}, name='{self.name}')"


class CameraManager:
    """Manages camera devices and provides unified capture interface."""
    
    def __init__(self, target_width: int = 640, target_height: int = 480):
        self.logger = get_logger()
        self.monitor = get_monitor()
        self.target_width = target_width
        self.target_height = target_height
        self.capture: Optional[cv2.VideoCapture] = None
        self.current_device: Optional[CameraDevice] = None
        self.scale_factor: float = 1.0
        self._lock = threading.Lock()
    
    def scan_devices(self, max_devices: int = 8) -> List[CameraDevice]:
        """
        Scan for available camera devices.
        
        Args:
            max_devices: Maximum number of devices to scan
            
        Returns:
            List of available CameraDevice objects sorted by resolution
        """
        self.logger.info(f"Scanning for camera devices (0-{max_devices-1})...")
        devices = []
        
        for i in range(max_devices):
            try:
                with timer_context(self.logger, f"Camera scan {i}", log_threshold=0.1):
                    cap = cv2.VideoCapture(i)
                    if not cap.isOpened():
                        continue
                    
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    
                    if width > 0 and height > 0:
                        device = CameraDevice(i, width, height)
                        devices.append(device)
                        self.logger.debug(f"Found {device}")
                    
                    cap.release()
                    
            except Exception as e:
                self.logger.warning(f"Error scanning camera {i}: {e}")
        
        # Sort by resolution (highest first)
        devices.sort(key=lambda d: d.resolution, reverse=True)
        
        self.logger.info(f"Found {len(devices)} camera device(s)")
        return devices
    
    def open_device(self, device_index: int = -1) -> bool:
        """
        Open a camera device for capture.
        
        Args:
            device_index: Specific device index, or -1 to auto-select best
            
        Returns:
            True if device opened successfully, False otherwise
        """
        with self._lock:
            # Close existing capture
            if self.capture is not None:
                self.close()
            
            # Select device
            if device_index == -1:
                devices = self.scan_devices()
                if not devices:
                    self.logger.error("No camera devices found")
                    return False
                self.current_device = devices[0]
                self.logger.info(f"Auto-selected best camera: {self.current_device}")
            else:
                try:
                    cap = cv2.VideoCapture(device_index)
                    if not cap.isOpened():
                        self.logger.error(f"Cannot open camera {device_index}")
                        return False
                    
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    self.current_device = CameraDevice(device_index, width, height)
                    cap.release()
                    
                except Exception as e:
                    self.logger.error(f"Error opening camera {device_index}: {e}")
                    return False
            
            # Open capture
            try:
                self.capture = cv2.VideoCapture(self.current_device.index)
                if not self.capture.isOpened():
                    self.logger.error(f"Failed to open capture for {self.current_device}")
                    self.capture = None
                    return False
                
                # Configure capture
                self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.current_device.width)
                self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.current_device.height)
                self.capture.set(cv2.CAP_PROP_FPS, 30)
                
                # Calculate scale factor
                self.scale_factor = min(
                    self.target_width / self.current_device.width,
                    self.target_height / self.current_device.height,
                    1.0
                )
                
                self.logger.info(
                    f"Opened {self.current_device} "
                    f"(scale: {self.scale_factor:.3f}, "
                    f"process: {self.target_width}x{self.target_height})"
                )
                return True
                
            except Exception as e:
                self.logger.error(f"Error configuring camera: {e}")
                self.close()
                return False
    
    @safe_execute("camera_capture", default_return=None, critical=True, context="frame capture")
    def capture_frame(self) -> Optional[np.ndarray]:
        """
        Capture a single frame from the camera.
        
        Returns:
            Captured frame as numpy array, or None if capture failed
        """
        if self.capture is None or not self.capture.isOpened():
            self.logger.error("Camera not opened")
            return None
        
        ret, frame = self.capture.read()
        if not ret or frame is None:
            self.logger.warning("Failed to read frame from camera")
            return None
        
        self.monitor.increment_counter("frames_captured")
        return frame
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Process a raw frame into RGB, grayscale, and optionally resized versions.
        
        Args:
            frame: Raw frame from camera (BGR format)
            
        Returns:
            Tuple of (rgb_frame, gray_frame, processed_frame)
        """
        if frame is None:
            raise ValueError("Cannot process None frame")
        
        # Resize if needed
        if self.scale_factor < 1.0:
            processed = cv2.resize(
                frame, 
                (self.target_width, self.target_height),
                interpolation=cv2.INTER_LINEAR
            )
        else:
            processed = frame
        
        # Convert color spaces
        rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        
        return rgb, gray, processed
    
    def get_intrinsics(self, fov_deg: float = 77.0) -> np.ndarray:
        """
        Get camera intrinsic matrix.
        
        Args:
            fov_deg: Horizontal field of view in degrees
            
        Returns:
            3x3 camera intrinsic matrix
        """
        if self.scale_factor < 1.0:
            w, h = self.target_width, self.target_height
        else:
            w, h = self.current_device.width, self.current_device.height
        
        fx = w / (2 * np.tan(np.radians(fov_deg) / 2))
        fy = fx  # Assume square pixels
        
        K = np.array([
            [fx, 0,  w / 2],
            [0,  fy, h / 2],
            [0,  0,     1  ]
        ], dtype=np.float64)
        
        return K
    
    def close(self):
        """Close the camera capture and release resources."""
        with self._lock:
            if self.capture is not None:
                self.capture.release()
                self.capture = None
                self.logger.info("Camera closed")
    
    def is_opened(self) -> bool:
        """Check if camera is currently opened."""
        return self.capture is not None and self.capture.isOpened()
    
    def get_info(self) -> dict:
        """Get camera information."""
        if self.current_device is None:
            return {"status": "not_opened"}
        
        return {
            "status": "opened",
            "device": {
                "index": self.current_device.index,
                "width": self.current_device.width,
                "height": self.current_device.height,
                "name": self.current_device.name,
            },
            "processing": {
                "target_width": self.target_width,
                "target_height": self.target_height,
                "scale_factor": self.scale_factor,
            }
        }
    
    def __del__(self):
        """Cleanup on deletion."""
        self.close()


if __name__ == "__main__":
    # Test the camera module
    from .logging import setup_logging
    
    logger = setup_logging(level="INFO")
    
    camera = CameraManager(target_width=640, target_height=480)
    
    # Scan devices
    devices = camera.scan_devices()
    print(f"Available devices: {devices}")
    
    # Open best device
    if camera.open_device():
        print(f"Camera info: {camera.get_info()}")
        
        # Capture a few frames
        for i in range(5):
            frame = camera.capture_frame()
            if frame is not None:
                rgb, gray, processed = camera.process_frame(frame)
                print(f"Frame {i}: {rgb.shape}, {gray.shape}")
        
        camera.close()
        print("Camera module test completed ✓")