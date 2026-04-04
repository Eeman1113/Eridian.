#!/usr/bin/env python3
"""
Enhanced Depth Estimation Module for Eridian
Supports multiple depth models with flexible registry system.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Type, Callable, Any
from abc import ABC, abstractmethod
import threading
from dataclasses import dataclass

from .logging import get_logger, get_monitor, safe_execute, timer_context


@dataclass
class ModelSpec:
    """Specification for a depth estimation model."""
    name: str
    display_name: str
    hub_source: str
    model_name: str
    transform_name: str
    fast: bool = False
    accurate: bool = False
    balanced: bool = False
    min_memory_mb: int = 500
    avg_latency_ms: float = 30.0


class BaseDepthEstimator(ABC):
    """Abstract base class for depth estimators."""
    
    def __init__(self, device: Optional[torch.device] = None,
                 max_depth: float = 8.0,
                 min_depth: float = 0.1):
        self.logger = get_logger()
        self.monitor = get_monitor()
        self.device = device or self._select_device()
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.model = None
        self.transform = None
        self._lock = threading.Lock()
    
    def _select_device(self) -> torch.device:
        """Auto-select appropriate device."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    
    @abstractmethod
    def _load_model(self):
        """Load the model - must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def _get_transform(self):
        """Get the appropriate transform - must be implemented by subclasses."""
        pass
    
    @torch.inference_mode()
    def estimate(self, rgb: np.ndarray) -> Optional[np.ndarray]:
        """Estimate depth from RGB image."""
        if self.model is None or self.transform is None:
            self.logger.error("Model not loaded")
            return None
        
        if rgb is None or rgb.size == 0:
            self.logger.warning("Invalid input image")
            return None
        
        with self._lock:
            try:
                with timer_context(self.logger, "Depth inference", log_threshold=0.05):
                    input_tensor = self.transform(rgb).to(self.device)
                    prediction = self.model(input_tensor)
                    prediction = F.interpolate(
                        prediction.unsqueeze(1),
                        size=rgb.shape[:2],
                        mode="bicubic",
                        align_corners=False
                    ).squeeze()
                    depth = prediction.cpu().numpy()
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
        """Estimate metric depth in meters."""
        depth_normalized = self.estimate(rgb)
        if depth_normalized is None:
            return None
        metric_depth = self.min_depth + (1.0 - depth_normalized) * (self.max_depth - self.min_depth)
        return metric_depth
    
    def get_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "device": str(self.device),
            "max_depth": self.max_depth,
            "min_depth": self.min_depth,
            "is_loaded": self.model is not None,
        }


class MiDaSEstimator(BaseDepthEstimator):
    """MiDaS-based depth estimator."""
    
    def __init__(self, model_spec: ModelSpec, **kwargs):
        self.spec = model_spec
        super().__init__(**kwargs)
        self._load_model()
    
    def _load_model(self):
        """Load MiDaS model from torch.hub."""
        try:
            self.logger.info(f"Loading {self.spec.display_name} on {self.device}...")
            with timer_context(self.logger, "Model loading", log_threshold=0.5):
                self.model = torch.hub.load(
                    self.spec.hub_source,
                    self.spec.model_name,
                    trust_repo=True
                )
                self.model.to(self.device).eval()
                transforms = torch.hub.load(
                    self.spec.hub_source,
                    self.spec.transform_name,
                    trust_repo=True
                )
                if "small" in self.spec.model_name.lower():
                    self.transform = transforms.small_transform
                else:
                    self.transform = transforms.dpt_transform
            self.logger.info(f"✓ {self.spec.display_name} loaded")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def get_info(self) -> Dict[str, Any]:
        info = super().get_info()
        info.update({
            "model_name": self.spec.display_name,
            "model_type": "MiDaS",
            "fast": self.spec.fast,
            "accurate": self.spec.accurate,
        })
        return info


class ModelRegistry:
    """Registry for depth estimation models."""
    
    _models: Dict[str, ModelSpec] = {}
    _estimators: Dict[str, Type[BaseDepthEstimator]] = {}
    
    @classmethod
    def register(cls, spec: ModelSpec, estimator_class: Type[BaseDepthEstimator]):
        """Register a model specification and its estimator class."""
        cls._models[spec.name] = spec
        cls._estimators[spec.name] = estimator_class
    
    @classmethod
    def get_model(cls, name: str) -> Optional[ModelSpec]:
        """Get model specification by name."""
        return cls._models.get(name)
    
    @classmethod
    def list_models(cls) -> list[ModelSpec]:
        """List all registered models."""
        return list(cls._models.values())
    
    @classmethod
    def list_model_names(cls) -> list[str]:
        """List all registered model names."""
        return list(cls._models.keys())
    
    @classmethod
    def get_fast_models(cls) -> list[ModelSpec]:
        """Get fast models."""
        return [m for m in cls._models.values() if m.fast]
    
    @classmethod
    def get_accurate_models(cls) -> list[ModelSpec]:
        """Get accurate models."""
        return [m for m in cls._models.values() if m.accurate]
    
    @classmethod
    def get_balanced_models(cls) -> list[ModelSpec]:
        """Get balanced models."""
        return [m for m in cls._models.values() if m.balanced]
    
    @classmethod
    def create_estimator(cls, name: str, **kwargs) -> Optional[BaseDepthEstimator]:
        """Create an estimator instance for a model."""
        spec = cls.get_model(name)
        if spec is None:
            raise ValueError(f"Model '{name}' not found")
        
        estimator_class = cls._estimators.get(name)
        if estimator_class is None:
            raise ValueError(f"No estimator registered for '{name}'")
        
        return estimator_class(spec, **kwargs)


# Register built-in MiDaS models
ModelRegistry.register(
    ModelSpec(
        name="midas_small",
        display_name="MiDaS Small",
        hub_source="intel-isl/MiDaS",
        model_name="MiDaS_small",
        transform_name="transforms",
        fast=True,
        balanced=True,
        min_memory_mb=200,
        avg_latency_ms=10.0,
    ),
    MiDaSEstimator
)

ModelRegistry.register(
    ModelSpec(
        name="midas_hybrid",
        display_name="MiDaS Hybrid",
        hub_source="intel-isl/MiDaS",
        model_name="DPT_Hybrid",
        transform_name="transforms",
        balanced=True,
        accurate=True,
        min_memory_mb=800,
        avg_latency_ms=50.0,
    ),
    MiDaSEstimator
)

ModelRegistry.register(
    ModelSpec(
        name="midas_large",
        display_name="MiDaS Large",
        hub_source="intel-isl/MiDaS",
        model_name="DPT_Large",
        transform_name="transforms",
        accurate=True,
        min_memory_mb=1500,
        avg_latency_ms=100.0,
    ),
    MiDaSEstimator
)


class MultiModelDepthEstimator:
    """
    Depth estimator that supports multiple models with automatic selection.
    """
    
    def __init__(self, preferred_model: str = "midas_small",
                 auto_select: bool = True,
                 **kwargs):
        """
        Initialize multi-model depth estimator.
        
        Args:
            preferred_model: Preferred model name
            auto_select: Automatically select best model based on hardware
            **kwargs: Additional arguments for estimator
        """
        self.logger = get_logger()
        self.monitor = get_monitor()
        
        self.preferred_model = preferred_model
        self.auto_select = auto_select
        self.estimator_kwargs = kwargs
        
        self.current_estimator: Optional[BaseDepthEstimator] = None
        self.current_model_name: Optional[str] = None
        
        self._select_model()
    
    def _select_model(self):
        """Select and load the appropriate model."""
        if self.auto_select:
            self.current_model_name = self._auto_select_model()
        else:
            self.current_model_name = self.preferred_model
        
        self.logger.info(f"Selected model: {self.current_model_name}")
        self._load_estimator()
    
    def _auto_select_model(self) -> str:
        """Automatically select the best model based on hardware."""
        # Check available memory
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
            self.logger.info(f"GPU memory: {gpu_memory:.2f} GB")
            
            if gpu_memory >= 8.0:
                return "midas_large"
            elif gpu_memory >= 4.0:
                return "midas_hybrid"
        
        # Check for Apple Silicon
        if torch.backends.mps.is_available():
            # MPS has unified memory, use hybrid if available
            return "midas_hybrid"
        
        # Default to small model for CPU
        return "midas_small"
    
    def _load_estimator(self):
        """Load the selected estimator."""
        try:
            self.current_estimator = ModelRegistry.create_estimator(
                self.current_model_name,
                **self.estimator_kwargs
            )
            self.logger.info(f"✓ Loaded {self.current_estimator.get_info()}")
        except Exception as e:
            self.logger.error(f"Failed to load estimator: {e}")
            raise
    
    def estimate(self, rgb: np.ndarray) -> Optional[np.ndarray]:
        """Estimate depth using current model."""
        if self.current_estimator is None:
            self.logger.error("No estimator loaded")
            return None
        return self.current_estimator.estimate(rgb)
    
    def estimate_metric(self, rgb: np.ndarray) -> Optional[np.ndarray]:
        """Estimate metric depth."""
        if self.current_estimator is None:
            return None
        return self.current_estimator.estimate_metric(rgb)
    
    def switch_model(self, model_name: str):
        """Switch to a different model."""
        if model_name not in ModelRegistry.list_model_names():
            raise ValueError(f"Model '{model_name}' not found")
        
        self.logger.info(f"Switching to model: {model_name}")
        self.current_model_name = model_name
        self._load_estimator()
    
    def get_available_models(self) -> list[ModelSpec]:
        """Get list of available models."""
        return ModelRegistry.list_models()
    
    def get_current_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        if self.current_estimator is None:
            return {}
        return self.current_estimator.get_info()
    
    def benchmark(self, rgb: np.ndarray, iterations: int = 10) -> Dict[str, float]:
        """
        Benchmark all available models.
        
        Args:
            rgb: Test image
            iterations: Number of iterations per model
            
        Returns:
            Dictionary mapping model names to average latency (ms)
        """
        results = {}
        
        for model_spec in ModelRegistry.list_models():
            try:
                estimator = ModelRegistry.create_estimator(
                    model_spec.name,
                    device=self.current_estimator.device if self.current_estimator else None,
                    max_depth=self.estimator_kwargs.get('max_depth', 8.0),
                    min_depth=self.estimator_kwargs.get('min_depth', 0.1),
                )
                
                latencies = []
                for _ in range(iterations):
                    import time
                    start = time.time()
                    estimator.estimate(rgb)
                    latencies.append((time.time() - start) * 1000)
                
                avg_latency = np.mean(latencies)
                results[model_spec.display_name] = avg_latency
                
                self.logger.info(
                    f"{model_spec.display_name}: {avg_latency:.1f}ms avg "
                    f"(min={min(latencies):.1f}ms, max={max(latencies):.1f}ms)"
                )
                
            except Exception as e:
                self.logger.warning(f"Failed to benchmark {model_spec.display_name}: {e}")
        
        return results


# Backwards compatibility aliases
DepthEstimator = MultiModelDepthEstimator
DepthEstimatorFactory = ModelRegistry


if __name__ == "__main__":
    # Test the enhanced depth estimation module
    from .logging import setup_logging
    import cv2
    
    logger = setup_logging(level="INFO")
    
    print("Available models:")
    for model in ModelRegistry.list_models():
        print(f"  - {model.display_name}: fast={model.fast}, accurate={model.accurate}")
    
    # Create multi-model estimator
    estimator = MultiModelDepthEstimator(preferred_model="midas_small")
    
    print(f"\nCurrent model: {estimator.get_current_model_info()}")
    
    # Test with camera if available
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
                
                # Benchmark (optional - can be slow)
                print("\nBenchmarking models...")
                benchmark_results = estimator.benchmark(rgb, iterations=3)
                print("\nBenchmark results:")
                for name, latency in sorted(benchmark_results.items(), key=lambda x: x[1]):
                    print(f"  {name}: {latency:.1f}ms")
        else:
            print("Could not capture test frame")
            
    except Exception as e:
        print(f"Test failed: {e}")
    
    print("\nEnhanced depth estimation module test completed ✓")