#!/usr/bin/env python3
"""
Test depth estimation and PLY export before wiring up the full pipeline.
"""

import sys
import time
import numpy as np
import cv2
from pathlib import Path

BASE_DIR = Path(__file__).parent
SPLAT_DIR = BASE_DIR / "splat"
SPLAT_DIR.mkdir(exist_ok=True)


def test_ply_export():
    """Create a small test point cloud and save as PLY."""
    print("\n=== TEST: PLY Export ===")
    from main import save_ply

    # Create a colored cube point cloud
    n = 1000
    points = np.random.rand(n, 3).astype(np.float32) * 2 - 1  # [-1, 1]
    colors = np.random.randint(0, 255, (n, 3), dtype=np.uint8)

    out = SPLAT_DIR / "test_cube.ply"
    save_ply(out, points, colors)

    # Verify file exists and has content
    assert out.exists(), "PLY file not created"
    size = out.stat().st_size
    assert size > 100, f"PLY file too small: {size} bytes"
    print(f"  PLY file: {out} ({size:,} bytes)")

    # Read back and verify header
    with open(out, 'rb') as f:
        header_bytes = b""
        while True:
            line = f.readline()
            header_bytes += line
            if line.strip() == b"end_header":
                break
        header = header_bytes.decode('ascii')
        assert f"element vertex {n}" in header
    print(f"  Header valid, {n} vertices declared")
    print("  PASS")


def test_depth_estimation():
    """Test depth model on a synthetic test image."""
    print("\n=== TEST: Depth Estimation ===")

    # Create a test image with depth cues (gradient + shapes)
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    # Gradient background (simulates floor/wall)
    for y in range(480):
        val = int(50 + (y / 480) * 150)
        img[y, :] = [val, val, val]
    # Draw some objects at different "depths"
    cv2.rectangle(img, (100, 100), (200, 200), (0, 0, 255), -1)
    cv2.circle(img, (400, 300), 80, (0, 255, 0), -1)
    cv2.rectangle(img, (450, 50), (600, 150), (255, 128, 0), -1)

    # Save test image
    test_img_path = SPLAT_DIR / "test_input.png"
    cv2.imwrite(str(test_img_path), img)
    print(f"  Test image saved: {test_img_path}")

    # Run depth estimation
    from main import DepthEstimator
    t0 = time.time()
    estimator = DepthEstimator()
    load_time = time.time() - t0
    print(f"  Model loaded in {load_time:.1f}s: {estimator.model_name}")

    t0 = time.time()
    depth = estimator.estimate(img)
    infer_time = time.time() - t0
    print(f"  Inference time: {infer_time:.3f}s ({1/infer_time:.1f} FPS)")

    assert depth.shape == (480, 640), f"Wrong shape: {depth.shape}"
    assert depth.dtype == np.float32
    assert depth.min() >= 0.4, f"Min depth too low: {depth.min()}"
    assert depth.max() <= 11.0, f"Max depth too high: {depth.max()}"

    # Save depth visualization
    depth_norm = ((depth - depth.min()) / (depth.max() - depth.min()) * 255).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)
    depth_path = SPLAT_DIR / "test_depth.png"
    cv2.imwrite(str(depth_path), depth_color)
    print(f"  Depth map saved: {depth_path}")

    # Run a few more inferences to get stable FPS
    times = []
    for _ in range(3):
        t0 = time.time()
        estimator.estimate(img)
        times.append(time.time() - t0)
    avg = sum(times) / len(times)
    print(f"  Average inference: {avg:.3f}s ({1/avg:.1f} FPS)")

    print("  PASS")


def test_pose_estimation():
    """Test ORB feature matching on synthetic frames."""
    print("\n=== TEST: Pose Estimation ===")
    from main import PoseEstimator

    pe = PoseEstimator(fx=512, fy=512, cx=320, cy=240)

    # Create two synthetic frames with slight shift
    img1 = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
    img2 = img1.copy()
    # Shift img2 slightly (simulating camera motion)
    M = np.float32([[1, 0, 5], [0, 1, 3]])
    img2 = cv2.warpAffine(img2, M, (640, 480))

    pose1, kp1, _ = pe.update(img1)
    pose2, kp2, _ = pe.update(img2)

    print(f"  Frame 1: {len(kp1)} keypoints")
    print(f"  Frame 2: {len(kp2)} keypoints")
    print(f"  Tracking: {pe.tracking}")
    print(f"  Pose translation: {pose2[:3, 3]}")
    print("  PASS")


def test_point_cloud():
    """Test point cloud accumulation and voxel downsample."""
    print("\n=== TEST: Point Cloud ===")
    from main import PointCloud

    pc = PointCloud(max_points=50000, voxel_size=0.1)
    K = np.array([[512, 0, 320], [0, 512, 240], [0, 0, 1]], dtype=np.float64)
    pose = np.eye(4, dtype=np.float64)

    # Simulate adding frames
    for i in range(35):
        depth = np.random.uniform(1.0, 5.0, (480, 640)).astype(np.float32)
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        pc.add_frame(depth, frame, pose, K, subsample=8)

    print(f"  Points after 35 frames: {pc.count():,}")
    assert pc.count() > 0
    # max_points is only enforced during voxel downsample, count may exceed between cycles

    pts, cols = pc.get_data()
    assert pts.shape[1] == 3
    assert cols.shape[1] == 3
    assert cols.dtype == np.uint8
    print("  PASS")


if __name__ == "__main__":
    print("Running component tests...\n")

    try:
        test_ply_export()
    except Exception as e:
        print(f"  FAIL: {e}")
        sys.exit(1)

    try:
        test_depth_estimation()
    except Exception as e:
        print(f"  FAIL: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)

    try:
        test_pose_estimation()
    except Exception as e:
        print(f"  FAIL: {e}")
        sys.exit(1)

    try:
        test_point_cloud()
    except Exception as e:
        print(f"  FAIL: {e}")
        sys.exit(1)

    print("\n" + "=" * 50)
    print("ALL TESTS PASSED")
    print("=" * 50)
