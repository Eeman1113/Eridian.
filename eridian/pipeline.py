#!/usr/bin/env python3
"""
Async Pipeline Module for Eridian
Manages asynchronous processing pipeline with thread pools and queues.
"""

import threading
import queue
import time
import numpy as np
from typing import Optional, Callable, Any, Dict, List
from dataclasses import dataclass, field
from enum import Enum
import concurrent.futures

from .logging import get_logger, get_monitor, safe_execute, timer_context


class PipelineStage(Enum):
    """Pipeline processing stages."""
    CAPTURE = "capture"
    DEPTH = "depth"
    POSE = "pose"
    SPLAT = "splat"
    SAVE = "save"
    VISUALIZE = "visualize"


@dataclass
class PipelineMetrics:
    """Metrics for pipeline performance monitoring."""
    stage_timings: Dict[PipelineStage, List[float]] = field(default_factory=dict)
    stage_counts: Dict[PipelineStage, int] = field(default_factory=dict)
    queue_sizes: Dict[str, List[int]] = field(default_factory=dict)
    total_frames: int = 0
    dropped_frames: int = 0
    start_time: float = field(default_factory=time.time)
    
    def record_timing(self, stage: PipelineStage, duration: float):
        """Record timing for a stage."""
        if stage not in self.stage_timings:
            self.stage_timings[stage] = []
        self.stage_timings[stage].append(duration)
        
        if stage not in self.stage_counts:
            self.stage_counts[stage] = 0
        self.stage_counts[stage] += 1
    
    def record_queue_size(self, queue_name: str, size: int):
        """Record queue size."""
        if queue_name not in self.queue_sizes:
            self.queue_sizes[queue_name] = []
        self.queue_sizes[queue_name].append(size)
    
    def get_stage_avg_time(self, stage: PipelineStage) -> Optional[float]:
        """Get average time for a stage."""
        if stage not in self.stage_timings or not self.stage_timings[stage]:
            return None
        return np.mean(self.stage_timings[stage])
    
    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        summary = {
            "total_frames": self.total_frames,
            "dropped_frames": self.dropped_frames,
            "runtime": time.time() - self.start_time,
            "fps": self.total_frames / (time.time() - self.start_time) if time.time() > self.start_time else 0,
            "stage_avg_times": {
                stage.value: self.get_stage_avg_time(stage)
                for stage in PipelineStage
            },
            "stage_counts": {
                stage.value: count
                for stage, count in self.stage_counts.items()
            },
        }
        return summary


@dataclass
class FrameData:
    """Data container for a frame in the pipeline."""
    frame_id: int
    rgb: Optional[np.ndarray] = None
    gray: Optional[np.ndarray] = None
    depth: Optional[np.ndarray] = None
    pose: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class AsyncPipeline:
    """
    Asynchronous pipeline manager for concurrent frame processing.
    Uses thread pools and queues for efficient multi-stage processing.
    """
    
    def __init__(self, max_workers: int = 4, 
                 queue_size: int = 10,
                 enable_visualization: bool = True):
        """
        Initialize async pipeline.
        
        Args:
            max_workers: Maximum number of worker threads
            queue_size: Maximum size for inter-stage queues
            enable_visualization: Enable visualization stage
        """
        self.logger = get_logger()
        self.monitor = get_monitor()
        
        self.max_workers = max_workers
        self.queue_size = queue_size
        self.enable_visualization = enable_visualization
        
        # Processing stages (callable functions)
        self.stages: Dict[PipelineStage, Callable] = {}
        
        # Thread pool executor
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="pipeline"
        )
        
        # Queues for inter-stage communication
        self.queues: Dict[str, queue.Queue] = {
            "input": queue.Queue(maxsize=queue_size),
            "depth": queue.Queue(maxsize=queue_size),
            "pose": queue.Queue(maxsize=queue_size),
            "splat": queue.Queue(maxsize=queue_size),
            "output": queue.Queue(maxsize=queue_size),
        }
        
        # Pipeline state
        self.running = False
        self.frame_counter = 0
        self.metrics = PipelineMetrics()
        
        # Worker threads
        self._workers: List[threading.Thread] = []
        
        self.logger.info(f"Async pipeline initialized (workers={max_workers}, queue_size={queue_size})")
    
    def register_stage(self, stage: PipelineStage, handler: Callable):
        """
        Register a processing stage handler.
        
        Args:
            stage: Pipeline stage identifier
            handler: Callable that processes FrameData
        """
        self.stages[stage] = handler
        self.logger.info(f"Registered stage: {stage.value}")
    
    def _depth_worker(self):
        """Worker thread for depth estimation."""
        while self.running or not self.queues["input"].empty():
            try:
                frame_data = self.queues["input"].get(timeout=0.1)
                
                start_time = time.time()
                
                # Process depth
                if PipelineStage.DEPTH in self.stages:
                    result = self.stages[PipelineStage.DEPTH](frame_data)
                    if result is not None:
                        frame_data = result
                
                duration = time.time() - start_time
                self.metrics.record_timing(PipelineStage.DEPTH, duration)
                
                # Pass to next stage
                self.queues["depth"].put(frame_data)
                self.queues["input"].task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Depth worker error: {e}")
    
    def _pose_worker(self):
        """Worker thread for pose estimation."""
        while self.running or not self.queues["depth"].empty():
            try:
                frame_data = self.queues["depth"].get(timeout=0.1)
                
                start_time = time.time()
                
                # Process pose
                if PipelineStage.POSE in self.stages:
                    result = self.stages[PipelineStage.POSE](frame_data)
                    if result is not None:
                        frame_data = result
                
                duration = time.time() - start_time
                self.metrics.record_timing(PipelineStage.POSE, duration)
                
                # Pass to next stage
                self.queues["pose"].put(frame_data)
                self.queues["depth"].task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Pose worker error: {e}")
    
    def _splat_worker(self):
        """Worker thread for splat building."""
        while self.running or not self.queues["pose"].empty():
            try:
                frame_data = self.queues["pose"].get(timeout=0.1)
                
                start_time = time.time()
                
                # Process splat
                if PipelineStage.SPLAT in self.stages:
                    result = self.stages[PipelineStage.SPLAT](frame_data)
                    if result is not None:
                        frame_data = result
                
                duration = time.time() - start_time
                self.metrics.record_timing(PipelineStage.SPLAT, duration)
                
                # Pass to next stage
                self.queues["splat"].put(frame_data)
                self.queues["pose"].task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Splat worker error: {e}")
    
    def _output_worker(self):
        """Worker thread for output processing (save/visualize)."""
        while self.running or not self.queues["splat"].empty():
            try:
                frame_data = self.queues["splat"].get(timeout=0.1)
                
                start_time = time.time()
                
                # Process save
                if PipelineStage.SAVE in self.stages:
                    self.stages[PipelineStage.SAVE](frame_data)
                
                # Process visualization
                if self.enable_visualization and PipelineStage.VISUALIZE in self.stages:
                    self.stages[PipelineStage.VISUALIZE](frame_data)
                
                duration = time.time() - start_time
                self.metrics.record_timing(PipelineStage.SAVE, duration)
                
                self.queues["output"].put(frame_data)
                self.queues["splat"].task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Output worker error: {e}")
    
    def start(self):
        """Start the pipeline workers."""
        if self.running:
            self.logger.warning("Pipeline already running")
            return
        
        self.running = True
        self.metrics = PipelineMetrics()
        
        # Start worker threads
        self._workers = [
            threading.Thread(target=self._depth_worker, name="depth_worker", daemon=True),
            threading.Thread(target=self._pose_worker, name="pose_worker", daemon=True),
            threading.Thread(target=self._splat_worker, name="splat_worker", daemon=True),
            threading.Thread(target=self._output_worker, name="output_worker", daemon=True),
        ]
        
        for worker in self._workers:
            worker.start()
        
        self.logger.info("Pipeline started")
    
    def stop(self, timeout: float = 5.0):
        """
        Stop the pipeline workers.
        
        Args:
            timeout: Maximum time to wait for workers to finish
        """
        if not self.running:
            return
        
        self.logger.info("Stopping pipeline...")
        self.running = False
        
        # Wait for workers to finish
        for worker in self._workers:
            worker.join(timeout=timeout)
        
        # Shutdown executor
        self.executor.shutdown(wait=True, timeout=timeout)
        
        self.logger.info("Pipeline stopped")
    
    @safe_execute("pipeline_submit", default_return=False, critical=False, context="frame submission")
    def submit_frame(self, rgb: np.ndarray, gray: np.ndarray, 
                    metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Submit a frame to the pipeline.
        
        Args:
            rgb: RGB image
            gray: Grayscale image
            metadata: Optional metadata dictionary
            
        Returns:
            True if frame submitted successfully, False if queue full
        """
        frame_data = FrameData(
            frame_id=self.frame_counter,
            rgb=rgb,
            gray=gray,
            metadata=metadata or {}
        )
        
        try:
            self.queues["input"].put(frame_data, block=False)
            self.frame_counter += 1
            self.metrics.total_frames += 1
            self.monitor.increment_counter("frames_submitted")
            return True
        except queue.Full:
            self.metrics.dropped_frames += 1
            self.logger.warning("Input queue full, dropping frame")
            return False
    
    def get_output(self, timeout: float = 0.1) -> Optional[FrameData]:
        """
        Get processed frame from output queue.
        
        Args:
            timeout: Maximum time to wait for output
            
        Returns:
            Processed frame data or None if queue empty
        """
        try:
            return self.queues["output"].get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_metrics(self) -> PipelineMetrics:
        """Get pipeline metrics."""
        return self.metrics
    
    def get_queue_sizes(self) -> Dict[str, int]:
        """Get current queue sizes."""
        return {
            name: q.qsize()
            for name, q in self.queues.items()
        }
    
    def is_running(self) -> bool:
        """Check if pipeline is running."""
        return self.running
    
    def __del__(self):
        """Cleanup on deletion."""
        self.stop(timeout=1.0)


class PipelineBuilder:
    """Builder for constructing async pipelines with chained stages."""
    
    def __init__(self):
        self.pipeline: Optional[AsyncPipeline] = None
        self._config: Dict[str, Any] = {}
    
    def with_workers(self, max_workers: int) -> 'PipelineBuilder':
        """Set maximum number of worker threads."""
        self._config['max_workers'] = max_workers
        return self
    
    def with_queue_size(self, queue_size: int) -> 'PipelineBuilder':
        """Set queue size for inter-stage communication."""
        self._config['queue_size'] = queue_size
        return self
    
    def with_visualization(self, enable: bool) -> 'PipelineBuilder':
        """Enable or disable visualization stage."""
        self._config['enable_visualization'] = enable
        return self
    
    def build(self) -> AsyncPipeline:
        """Build the async pipeline."""
        self.pipeline = AsyncPipeline(**self._config)
        return self.pipeline


if __name__ == "__main__":
    # Test the async pipeline
    from .logging import setup_logging
    
    logger = setup_logging(level="INFO")
    
    # Create pipeline
    pipeline = PipelineBuilder() \
        .with_workers(4) \
        .with_queue_size(5) \
        .with_visualization(False) \
        .build()
    
    # Register dummy stages
    def dummy_depth(frame_data):
        time.sleep(0.01)  # Simulate processing
        frame_data.depth = np.random.rand(480, 640).astype(np.float32)
        return frame_data
    
    def dummy_pose(frame_data):
        time.sleep(0.01)  # Simulate processing
        frame_data.pose = np.eye(4, dtype=np.float32)
        return frame_data
    
    def dummy_splat(frame_data):
        time.sleep(0.01)  # Simulate processing
        return frame_data
    
    pipeline.register_stage(PipelineStage.DEPTH, dummy_depth)
    pipeline.register_stage(PipelineStage.POSE, dummy_pose)
    pipeline.register_stage(PipelineStage.SPLAT, dummy_splat)
    
    # Start pipeline
    pipeline.start()
    
    # Submit some frames
    print("Submitting frames...")
    for i in range(20):
        rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        gray = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        
        if not pipeline.submit_frame(rgb, gray, {"frame_num": i}):
            print(f"Frame {i} dropped")
        else:
            print(f"Frame {i} submitted")
        
        time.sleep(0.05)
    
    # Wait for processing
    time.sleep(2.0)
    
    # Get metrics
    metrics = pipeline.get_metrics()
    summary = metrics.get_summary()
    print(f"\nPipeline Summary:")
    print(f"  Total frames: {summary['total_frames']}")
    print(f"  Dropped frames: {summary['dropped_frames']}")
    print(f"  FPS: {summary['fps']:.2f}")
    print(f"  Stage timings:")
    for stage, avg_time in summary['stage_avg_times'].items():
        if avg_time:
            print(f"    {stage}: {avg_time*1000:.1f}ms")
    
    # Stop pipeline
    pipeline.stop()
    print("Async pipeline test completed ✓")