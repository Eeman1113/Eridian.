#!/usr/bin/env python3
"""
Configuration Manager for Eridian
Handles loading, validation, and access to configuration settings.
"""

import yaml
import sys
from pathlib import Path
from typing import Any, Dict, Optional
import dataclasses


@dataclasses.dataclass
class CameraConfig:
    width: int = 640
    height: int = 480
    fov_deg: float = 77.0
    target_fps: int = 15


@dataclasses.dataclass
class DepthConfig:
    model: str = "MiDaS_small"
    min_depth: float = 0.1
    max_depth: float = 8.0
    min_disparity: float = 0.05
    max_disparity: float = 0.92


@dataclasses.dataclass
class SplatConfig:
    max_points: int = 800000
    point_step: int = 4
    output_dir: str = "./splat"


@dataclasses.dataclass
class PerformanceConfig:
    save_interval: int = 30
    backup_interval: int = 150
    downsample_ratio: float = 0.8
    downsample_interval: int = 8


@dataclasses.dataclass
class SLAMConfig:
    max_features: int = 500
    feature_quality: float = 0.03
    feature_min_distance: int = 7
    feature_refresh_interval: int = 20
    ransac_threshold: float = 1.0


@dataclasses.dataclass
class VisualizationConfig:
    show_depth_by_default: bool = False
    viewer_update_interval: int = 30
    rerun_app_name: str = "eridian"
    rerun_spawn: bool = True


@dataclasses.dataclass
class LoggingConfig:
    level: str = "INFO"
    console: bool = True
    fps_update_interval: int = 20


@dataclasses.dataclass
class AdvancedConfig:
    camera_index: int = -1
    enable_cuda: bool = True
    enable_mps: bool = True
    worker_threads: int = 0


class Config:
    """Main configuration class that aggregates all sub-configurations."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.camera = CameraConfig()
        self.depth = DepthConfig()
        self.splat = SplatConfig()
        self.performance = PerformanceConfig()
        self.slam = SLAMConfig()
        self.visualization = VisualizationConfig()
        self.logging = LoggingConfig()
        self.advanced = AdvancedConfig()
        
        if config_path:
            self.load_from_file(config_path)
        else:
            self.load_default()
    
    def load_from_file(self, config_path: str):
        """Load configuration from a YAML file."""
        try:
            config_file = Path(config_path)
            if not config_file.exists():
                print(f"[Config] Warning: Config file not found: {config_path}")
                print(f"[Config] Using default configuration")
                return
            
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f)
            
            self._apply_config(config_data)
            print(f"[Config] ✓ Loaded configuration from {config_path}")
            
        except Exception as e:
            print(f"[Config] ✗ Error loading config: {e}")
            print(f"[Config] Using default configuration")
    
    def load_default(self):
        """Load default configuration from config/default.yaml."""
        default_path = Path(__file__).parent.parent / "config" / "default.yaml"
        if default_path.exists():
            self.load_from_file(str(default_path))
        else:
            print(f"[Config] Using built-in defaults")
    
    def _apply_config(self, config_data: Dict[str, Any]):
        """Apply configuration data to config objects."""
        if not config_data:
            return
        
        # Camera settings
        if 'camera' in config_data:
            cam = config_data['camera']
            self.camera.width = cam.get('width', self.camera.width)
            self.camera.height = cam.get('height', self.camera.height)
            self.camera.fov_deg = cam.get('fov_deg', self.camera.fov_deg)
            self.camera.target_fps = cam.get('target_fps', self.camera.target_fps)
        
        # Depth settings
        if 'depth' in config_data:
            depth = config_data['depth']
            self.depth.model = depth.get('model', self.depth.model)
            self.depth.min_depth = depth.get('min_depth', self.depth.min_depth)
            self.depth.max_depth = depth.get('max_depth', self.depth.max_depth)
            self.depth.min_disparity = depth.get('min_disparity', self.depth.min_disparity)
            self.depth.max_disparity = depth.get('max_disparity', self.depth.max_disparity)
        
        # Splat settings
        if 'splat' in config_data:
            splat = config_data['splat']
            self.splat.max_points = splat.get('max_points', self.splat.max_points)
            self.splat.point_step = splat.get('point_step', self.splat.point_step)
            self.splat.output_dir = splat.get('output_dir', self.splat.output_dir)
        
        # Performance settings
        if 'performance' in config_data:
            perf = config_data['performance']
            self.performance.save_interval = perf.get('save_interval', self.performance.save_interval)
            self.performance.backup_interval = perf.get('backup_interval', self.performance.backup_interval)
            self.performance.downsample_ratio = perf.get('downsample_ratio', self.performance.downsample_ratio)
            self.performance.downsample_interval = perf.get('downsample_interval', self.performance.downsample_interval)
        
        # SLAM settings
        if 'slam' in config_data:
            slam = config_data['slam']
            self.slam.max_features = slam.get('max_features', self.slam.max_features)
            self.slam.feature_quality = slam.get('feature_quality', self.slam.feature_quality)
            self.slam.feature_min_distance = slam.get('feature_min_distance', self.slam.feature_min_distance)
            self.slam.feature_refresh_interval = slam.get('feature_refresh_interval', self.slam.feature_refresh_interval)
            self.slam.ransac_threshold = slam.get('ransac_threshold', self.slam.ransac_threshold)
        
        # Visualization settings
        if 'visualization' in config_data:
            vis = config_data['visualization']
            self.visualization.show_depth_by_default = vis.get('show_depth_by_default', self.visualization.show_depth_by_default)
            self.visualization.viewer_update_interval = vis.get('viewer_update_interval', self.visualization.viewer_update_interval)
            self.visualization.rerun_app_name = vis.get('rerun_app_name', self.visualization.rerun_app_name)
            self.visualization.rerun_spawn = vis.get('rerun_spawn', self.visualization.rerun_spawn)
        
        # Logging settings
        if 'logging' in config_data:
            log = config_data['logging']
            self.logging.level = log.get('level', self.logging.level)
            self.logging.console = log.get('console', self.logging.console)
            self.logging.fps_update_interval = log.get('fps_update_interval', self.logging.fps_update_interval)
        
        # Advanced settings
        if 'advanced' in config_data:
            adv = config_data['advanced']
            self.advanced.camera_index = adv.get('camera_index', self.advanced.camera_index)
            self.advanced.enable_cuda = adv.get('enable_cuda', self.advanced.enable_cuda)
            self.advanced.enable_mps = adv.get('enable_mps', self.advanced.enable_mps)
            self.advanced.worker_threads = adv.get('worker_threads', self.advanced.worker_threads)
    
    def validate(self) -> bool:
        """Validate configuration settings."""
        errors = []
        
        if self.camera.width <= 0 or self.camera.height <= 0:
            errors.append("Camera dimensions must be positive")
        
        if self.camera.fov_deg <= 0 or self.camera.fov_deg > 180:
            errors.append("Camera FOV must be between 0 and 180 degrees")
        
        if self.depth.min_depth >= self.depth.max_depth:
            errors.append("min_depth must be less than max_depth")
        
        if self.splat.max_points <= 0:
            errors.append("max_points must be positive")
        
        if self.splat.point_step < 1:
            errors.append("point_step must be at least 1")
        
        if self.performance.save_interval <= 0:
            errors.append("save_interval must be positive")
        
        if errors:
            print("[Config] ✗ Validation errors:")
            for error in errors:
                print(f"  - {error}")
            return False
        
        return True
    
    def print_summary(self):
        """Print a summary of the current configuration."""
        print("\n" + "="*60)
        print("Configuration Summary")
        print("="*60)
        print(f"Camera: {self.camera.width}x{self.camera.height} @ {self.camera.target_fps} FPS")
        print(f"Depth Model: {self.depth.model}")
        print(f"Depth Range: {self.depth.min_depth}m - {self.depth.max_depth}m")
        print(f"Max Points: {self.splat.max_points:,}")
        print(f"Point Step: {self.splat.point_step}")
        print(f"Output Dir: {self.splat.output_dir}")
        print(f"Save Interval: {self.performance.save_interval} frames")
        print("="*60 + "\n")


# Global configuration instance
_config: Optional[Config] = None


def get_config(config_path: Optional[str] = None) -> Config:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = Config(config_path)
        if not _config.validate():
            print("[Config] Using default configuration due to validation errors")
            _config = Config()
    return _config


def reset_config():
    """Reset the global configuration instance."""
    global _config
    _config = None


if __name__ == "__main__":
    # Test the configuration module
    config = get_config()
    config.print_summary()
    print("Configuration test passed ✓")