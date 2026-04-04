#!/usr/bin/env python3
"""
Unit Tests for Eridian Core Components
Basic test suite for core functionality.
"""

import unittest
import numpy as np
import tempfile
import shutil
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestConfig(unittest.TestCase):
    """Test configuration management."""
    
    def setUp(self):
        from eridian import reset_config
        reset_config()
    
    def test_default_config(self):
        """Test loading default configuration."""
        from eridian import get_config
        
        config = get_config()
        self.assertIsNotNone(config)
        self.assertEqual(config.camera.width, 640)
        self.assertEqual(config.camera.height, 480)
        self.assertEqual(config.depth.model, "MiDaS_small")
    
    def test_config_validation(self):
        """Test configuration validation."""
        from eridian import Config
        
        config = Config()
        self.assertTrue(config.validate())
    
    def test_config_summary(self):
        """Test configuration summary printing."""
        from eridian import get_config
        from io import StringIO
        import sys as _sys
        
        config = get_config()
        
        # Capture output
        old_stdout = _sys.stdout
        _sys.stdout = StringIO()
        
        config.print_summary()
        
        output = _sys.stdout.getvalue()
        _sys.stdout = old_stdout
        
        self.assertIn("Configuration Summary", output)
        self.assertIn("Camera:", output)


class TestLogging(unittest.TestCase):
    """Test logging system."""
    
    def test_logger_creation(self):
        """Test logger instance creation."""
        from eridian import setup_logging, get_logger
        
        logger = setup_logging(level="DEBUG", console=False)
        self.assertIsNotNone(logger)
        
        # Test singleton behavior
        logger2 = get_logger()
        self.assertEqual(logger, logger2)
    
    def test_performance_monitor(self):
        """Test performance monitoring."""
        from eridian import get_monitor
        
        monitor = get_monitor()
        
        # Test timer
        monitor.start_timer("test_timer")
        import time
        time.sleep(0.01)
        duration = monitor.stop_timer("test_timer")
        self.assertGreater(duration, 0.01)
        
        # Test counter
        monitor.increment_counter("test_counter")
        count = monitor.get_counter("test_counter")
        self.assertEqual(count, 1)
    
    def test_error_handler(self):
        """Test error handling."""
        from eridian import get_error_handler
        
        handler = get_error_handler()
        
        # Test error handling
        test_error = ValueError("Test error")
        result = handler.handle_error("test_type", test_error, "test_context", critical=False)
        self.assertTrue(result)  # Should continue


class TestSpatialHash(unittest.TestCase):
    """Test spatial hashing."""
    
    def test_spatial_hash_creation(self):
        """Test spatial hash creation."""
        from eridian import SpatialHash
        
        sh = SpatialHash(cell_size=0.5)
        self.assertEqual(sh.cell_size, 0.5)
    
    def test_point_insertion(self):
        """Test point insertion."""
        from eridian import SpatialHash
        
        sh = SpatialHash(cell_size=0.5)
        pos = np.array([1.0, 2.0, 3.0])
        idx = sh.insert(pos)
        self.assertEqual(idx, 0)
    
    def test_batch_insertion(self):
        """Test batch point insertion."""
        from eridian import SpatialHash
        
        sh = SpatialHash(cell_size=0.5)
        positions = np.random.randn(100, 3) * 5.0
        indices = sh.insert_batch(positions)
        self.assertEqual(len(indices), 100)
    
    def test_radius_query(self):
        """Test radius query."""
        from eridian import SpatialHash
        
        sh = SpatialHash(cell_size=0.5)
        
        # Insert points around origin
        for i in range(10):
            pos = np.random.randn(3) * 1.0
            sh.insert(pos)
        
        # Query near origin
        neighbors = sh.query_radius(np.zeros(3), radius=2.0)
        self.assertGreater(len(neighbors), 0)
    
    def test_statistics(self):
        """Test statistics reporting."""
        from eridian import SpatialHash
        
        sh = SpatialHash(cell_size=0.5)
        positions = np.random.randn(100, 3) * 5.0
        sh.insert_batch(positions)
        
        stats = sh.get_stats()
        self.assertEqual(stats['num_points'], 100)
        self.assertGreater(stats['occupied_cells'], 0)


class TestOctree(unittest.TestCase):
    """Test octree implementation."""
    
    def test_octree_creation(self):
        """Test octree creation."""
        from eridian import Octree
        
        octree = Octree(size=10.0, max_points_per_node=10)
        self.assertEqual(octree.total_points, 0)
    
    def test_point_insertion(self):
        """Test point insertion."""
        from eridian import Octree
        
        octree = Octree(size=10.0, max_points_per_node=10)
        pos = np.array([1.0, 2.0, 3.0])
        color = np.array([0.5, 0.5, 0.5])
        
        result = octree.insert(pos, color)
        self.assertTrue(result)
        self.assertEqual(octree.total_points, 1)
    
    def test_radius_query(self):
        """Test radius query."""
        from eridian import Octree
        
        octree = Octree(size=10.0, max_points_per_node=100)
        
        # Insert points
        for i in range(50):
            pos = np.random.randn(3) * 2.0
            color = np.random.rand(3)
            octree.insert(pos, color)
        
        # Query
        results = octree.query_radius(np.zeros(3), radius=3.0)
        self.assertGreater(len(results), 0)
    
    def test_get_all_points(self):
        """Test retrieving all points."""
        from eridian import Octree
        
        octree = Octree(size=10.0, max_points_per_node=10)
        
        positions = np.random.randn(20, 3) * 3.0
        colors = np.random.rand(20, 3)
        
        for i, (pos, col) in enumerate(zip(positions, colors)):
            octree.insert(pos, col, index=i)
        
        all_pos, all_col = octree.get_all_points()
        self.assertEqual(len(all_pos), 20)
        self.assertIsNotNone(all_col)


class TestSplatBuilder(unittest.TestCase):
    """Test splat builder functionality."""
    
    def setUp(self):
        """Create temporary directory for tests."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)
    
    def test_splat_builder_creation(self):
        """Test splat builder creation."""
        from eridian import SplatBuilder
        
        builder = SplatBuilder(max_points=1000, point_step=4, output_dir=self.temp_dir)
        self.assertEqual(builder.max_points, 1000)
    
    def test_add_points(self):
        """Test adding points to splat."""
        from eridian import SplatBuilder
        
        builder = SplatBuilder(max_points=1000, output_dir=self.temp_dir)
        
        # Create dummy data
        rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        depth = np.random.rand(480, 640).astype(np.float32)
        pose = np.eye(4, dtype=np.float64)
        K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64)
        
        # Add points
        builder.add(rgb, depth, pose, K)
        
        stats = builder.get_stats()
        self.assertGreater(stats['total_points'], 0)
    
    def test_get_arrays(self):
        """Test getting point arrays."""
        from eridian import SplatBuilder
        
        builder = SplatBuilder(max_points=1000, output_dir=self.temp_dir)
        
        rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        depth = np.random.rand(480, 640).astype(np.float32)
        pose = np.eye(4, dtype=np.float64)
        K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64)
        
        builder.add(rgb, depth, pose, K)
        
        positions, colors = builder.get_arrays()
        self.assertIsNotNone(positions)
        self.assertIsNotNone(colors)
        self.assertEqual(len(positions), len(colors))
    
    def test_save_ply(self):
        """Test saving PLY file."""
        from eridian import SplatBuilder
        
        builder = SplatBuilder(max_points=1000, output_dir=self.temp_dir)
        
        rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        depth = np.random.rand(480, 640).astype(np.float32)
        pose = np.eye(4, dtype=np.float64)
        K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64)
        
        builder.add(rgb, depth, pose, K)
        
        # Save
        success = builder.save("test.ply")
        self.assertTrue(success)
        
        # Check file exists
        ply_path = Path(self.temp_dir) / "test.ply"
        self.assertTrue(ply_path.exists())
    
    def test_memory_estimation(self):
        """Test memory usage estimation."""
        from eridian import SplatBuilder
        
        builder = SplatBuilder(max_points=1000, output_dir=self.temp_dir)
        
        rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        depth = np.random.rand(480, 640).astype(np.float32)
        pose = np.eye(4, dtype=np.float64)
        K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64)
        
        builder.add(rgb, depth, pose, K)
        
        memory_mb = builder.estimate_memory()
        self.assertGreater(memory_mb, 0)


class TestModelRegistry(unittest.TestCase):
    """Test model registry functionality."""
    
    def test_list_models(self):
        """Test listing available models."""
        from eridian import ModelRegistry
        
        models = ModelRegistry.list_models()
        self.assertGreater(len(models), 0)
    
    def test_get_model(self):
        """Test getting specific model."""
        from eridian import ModelRegistry
        
        model = ModelRegistry.get_model("midas_small")
        self.assertIsNotNone(model)
        self.assertEqual(model.name, "midas_small")
    
    def test_fast_models(self):
        """Test getting fast models."""
        from eridian import ModelRegistry
        
        fast_models = ModelRegistry.get_fast_models()
        self.assertGreater(len(fast_models), 0)
        
        for model in fast_models:
            self.assertTrue(model.fast)
    
    def test_accurate_models(self):
        """Test getting accurate models."""
        from eridian import ModelRegistry
        
        accurate_models = ModelRegistry.get_accurate_models()
        self.assertGreater(len(accurate_models), 0)
        
        for model in accurate_models:
            self.assertTrue(model.accurate)


class TestPipeline(unittest.TestCase):
    """Test async pipeline functionality."""
    
    def test_pipeline_creation(self):
        """Test pipeline creation."""
        from eridian import AsyncPipeline
        
        pipeline = AsyncPipeline(max_workers=2, queue_size=5)
        self.assertEqual(pipeline.max_workers, 2)
        self.assertEqual(pipeline.queue_size, 5)
    
    def test_pipeline_builder(self):
        """Test pipeline builder."""
        from eridian import PipelineBuilder
        
        pipeline = PipelineBuilder() \
            .with_workers(4) \
            .with_queue_size(10) \
            .build()
        
        self.assertEqual(pipeline.max_workers, 4)
        self.assertEqual(pipeline.queue_size, 10)
    
    def test_submit_frame(self):
        """Test frame submission."""
        from eridian import AsyncPipeline
        
        pipeline = AsyncPipeline(max_workers=2, queue_size=5)
        
        rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        gray = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        
        success = pipeline.submit_frame(rgb, gray)
        self.assertTrue(success)
    
    def test_metrics(self):
        """Test pipeline metrics."""
        from eridian import AsyncPipeline
        
        pipeline = AsyncPipeline(max_workers=2, queue_size=5)
        
        rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        gray = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        
        # Submit frames
        for i in range(5):
            pipeline.submit_frame(rgb, gray)
        
        metrics = pipeline.get_metrics()
        self.assertEqual(metrics.total_frames, 5)


def run_tests():
    """Run all tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestConfig))
    suite.addTests(loader.loadTestsFromTestCase(TestLogging))
    suite.addTests(loader.loadTestsFromTestCase(TestSpatialHash))
    suite.addTests(loader.loadTestsFromTestCase(TestOctree))
    suite.addTests(loader.loadTestsFromTestCase(TestSplatBuilder))
    suite.addTests(loader.loadTestsFromTestCase(TestModelRegistry))
    suite.addTests(loader.loadTestsFromTestCase(TestPipeline))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return exit code
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_tests())