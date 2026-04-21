"""
GCT Backend for Eridian — Geometric Context Transformer integration.

Uses the LingBot-MAP GCT model (https://github.com/Robbyant/lingbot-map)
for feed-forward streaming 3D reconstruction. Replaces the separate
Depth Anything V2 + Lucas-Kanade + PnP pipeline with a single transformer
that jointly predicts camera poses, depth, and 3D world points.

Credits:
    LingBot-MAP: Geometric Context Transformer for Streaming 3D Reconstruction
    Paper: arXiv:2604.14141 (2026)
    Code: https://github.com/Robbyant/lingbot-map
    License: Apache 2.0
    Authors: Robbyant Team
"""

import logging
import os
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as TF

log = logging.getLogger("mapper")

# ---------------------------------------------------------------------------
# Device / dtype helpers for Apple Silicon MPS
# ---------------------------------------------------------------------------

def get_device_and_dtype():
    """Select best available device and dtype for GCT inference."""
    if torch.backends.mps.is_available():
        return torch.device("mps"), torch.float32
    elif torch.cuda.is_available():
        cap = torch.cuda.get_device_capability()[0]
        dtype = torch.bfloat16 if cap >= 8 else torch.float16
        return torch.device("cuda"), dtype
    return torch.device("cpu"), torch.float32


# ---------------------------------------------------------------------------
# Image preprocessing (adapted from lingbot_map.utils.load_fn)
# ---------------------------------------------------------------------------

def preprocess_frame(frame_bgr: np.ndarray, image_size: int = 518,
                     patch_size: int = 14) -> torch.Tensor:
    """Preprocess a single BGR OpenCV frame into a model-ready tensor.

    Returns [1, 3, H, W] float32 tensor in [0, 1] range.
    """
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    width, height = img.size

    new_width = image_size
    new_height = round(height * (new_width / width) / patch_size) * patch_size
    img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)

    to_tensor = TF.ToTensor()
    tensor = to_tensor(img)  # [3, H, W] in [0, 1]

    # Center crop height if needed
    if new_height > image_size:
        start_y = (new_height - image_size) // 2
        tensor = tensor[:, start_y:start_y + image_size, :]

    return tensor.unsqueeze(0)  # [1, 3, H, W]


def preprocess_frames_batch(frames_bgr: list, image_size: int = 518,
                            patch_size: int = 14) -> torch.Tensor:
    """Preprocess a list of BGR frames into a batch tensor [N, 3, H, W]."""
    tensors = [preprocess_frame(f, image_size, patch_size) for f in frames_bgr]
    return torch.cat(tensors, dim=0)


# ---------------------------------------------------------------------------
# GCT Model wrapper — MPS-compatible
# ---------------------------------------------------------------------------

class GCTReconstructor:
    """Wraps LingBot-MAP's GCTStream for use in Eridian on Apple M-series.

    This replaces DepthEstimator + PoseEstimator with a single model
    that predicts camera pose, depth, and 3D world points jointly.

    Credits:
        LingBot-MAP by Robbyant Team
        https://github.com/Robbyant/lingbot-map
        Apache 2.0 License
    """

    def __init__(self, model_path: str, image_size: int = 518,
                 num_scale_frames: int = 8, conf_threshold: float = 1.5,
                 kv_cache_sliding_window: int = 64):
        self.image_size = image_size
        self.patch_size = 14
        self.num_scale_frames = num_scale_frames
        self.conf_threshold = conf_threshold

        self.device, self.dtype = get_device_and_dtype()
        # Force SDPA on non-CUDA (FlashInfer is CUDA-only)
        use_sdpa = self.device.type != "cuda"

        log.info(f"GCT device={self.device}, dtype={self.dtype}, sdpa={use_sdpa}")

        from eridian.lingbot_map.models.gct_stream import GCTStream

        self.model = GCTStream(
            img_size=image_size,
            patch_size=self.patch_size,
            enable_3d_rope=True,
            max_frame_num=1024,
            kv_cache_sliding_window=kv_cache_sliding_window,
            kv_cache_scale_frames=num_scale_frames,
            kv_cache_cross_frame_special=True,
            kv_cache_include_scale_frames=True,
            use_sdpa=use_sdpa,
            camera_num_iterations=4,
        )

        if model_path:
            log.info(f"Loading GCT checkpoint: {model_path}")
            ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
            state_dict = ckpt.get("model", ckpt)
            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
            if missing:
                log.info(f"  Missing keys: {len(missing)}")
            if unexpected:
                log.info(f"  Unexpected keys: {len(unexpected)}")

        self.model = self.model.to(self.device).eval()

        # Cast aggregator to inference dtype (saves memory, no quality loss)
        if self.dtype != torch.float32 and hasattr(self.model, "aggregator"):
            self.model.aggregator = self.model.aggregator.to(dtype=self.dtype)

    @torch.no_grad()
    def reconstruct_video(self, video_path: str, fps: int = 10,
                          max_frames: Optional[int] = None,
                          keyframe_interval: int = 1):
        """Run GCT streaming inference on a video file.

        Returns dict with numpy arrays: world_points, world_points_conf,
        depth, depth_conf, extrinsic, intrinsic, images.
        """
        from eridian.lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri
        from eridian.lingbot_map.utils.geometry import closed_form_inverse_se3_general

        # Extract frames
        frames = self._extract_video_frames(video_path, fps, max_frames)
        log.info(f"GCT: Processing {len(frames)} frames from {video_path}")

        images = preprocess_frames_batch(frames, self.image_size, self.patch_size)
        images = images.to(self.device)

        num_frames = images.shape[0]
        if keyframe_interval is None:
            keyframe_interval = max(1, (num_frames + 319) // 320) if num_frames > 320 else 1

        # Streaming inference — autocast only on CUDA (MPS doesn't need it with fp32)
        use_autocast = self.device.type == "cuda"
        ctx = torch.amp.autocast("cuda", dtype=self.dtype) if use_autocast else torch.inference_mode()
        with ctx:
            predictions = self.model.inference_streaming(
                images,
                num_scale_frames=min(self.num_scale_frames, num_frames),
                keyframe_interval=keyframe_interval,
                output_device=torch.device("cpu"),
            )

        # Post-process: pose encoding → extrinsics (c2w)
        pose_enc = predictions["pose_enc"]
        extrinsic, intrinsic = pose_encoding_to_extri_intri(
            pose_enc, images.shape[-2:]
        )

        # w2c → c2w
        ext_4x4 = torch.zeros((*extrinsic.shape[:-2], 4, 4),
                               device=extrinsic.device, dtype=extrinsic.dtype)
        ext_4x4[..., :3, :4] = extrinsic
        ext_4x4[..., 3, 3] = 1.0
        ext_4x4 = closed_form_inverse_se3_general(ext_4x4)
        extrinsic = ext_4x4[..., :3, :4]

        # Squeeze batch dim and move to numpy
        def to_np(t):
            if t.dim() > 0 and t.shape[0] == 1:
                t = t[0]
            return t.detach().cpu().numpy()

        result = {
            "world_points": to_np(predictions["world_points"]),
            "world_points_conf": to_np(predictions["world_points_conf"]),
            "depth": to_np(predictions["depth"]),
            "depth_conf": to_np(predictions["depth_conf"]),
            "extrinsic": to_np(extrinsic),
            "intrinsic": to_np(intrinsic),
            "images": to_np(predictions["images"]),
            "num_frames": num_frames,
        }
        return result

    @torch.no_grad()
    def reconstruct_frames(self, frames_bgr: list,
                           keyframe_interval: int = 1):
        """Run GCT streaming inference on a list of BGR frames.

        Returns dict with numpy arrays.
        """
        from eridian.lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri
        from eridian.lingbot_map.utils.geometry import closed_form_inverse_se3_general

        images = preprocess_frames_batch(frames_bgr, self.image_size, self.patch_size)
        images = images.to(self.device)

        num_frames = images.shape[0]
        if keyframe_interval is None:
            keyframe_interval = max(1, (num_frames + 319) // 320) if num_frames > 320 else 1

        use_autocast = self.device.type == "cuda"
        ctx = torch.amp.autocast("cuda", dtype=self.dtype) if use_autocast else torch.inference_mode()
        with ctx:
            predictions = self.model.inference_streaming(
                images,
                num_scale_frames=min(self.num_scale_frames, num_frames),
                keyframe_interval=keyframe_interval,
                output_device=torch.device("cpu"),
            )

        # Post-process
        pose_enc = predictions["pose_enc"]
        extrinsic, intrinsic = pose_encoding_to_extri_intri(
            pose_enc, images.shape[-2:]
        )

        ext_4x4 = torch.zeros((*extrinsic.shape[:-2], 4, 4),
                               device=extrinsic.device, dtype=extrinsic.dtype)
        ext_4x4[..., :3, :4] = extrinsic
        ext_4x4[..., 3, 3] = 1.0
        ext_4x4 = closed_form_inverse_se3_general(ext_4x4)
        extrinsic = ext_4x4[..., :3, :4]

        def to_np(t):
            if t.dim() > 0 and t.shape[0] == 1:
                t = t[0]
            return t.detach().cpu().numpy()

        return {
            "world_points": to_np(predictions["world_points"]),
            "world_points_conf": to_np(predictions["world_points_conf"]),
            "depth": to_np(predictions["depth"]),
            "depth_conf": to_np(predictions["depth_conf"]),
            "extrinsic": to_np(extrinsic),
            "intrinsic": to_np(intrinsic),
            "images": to_np(predictions["images"]),
            "num_frames": num_frames,
        }

    def _extract_video_frames(self, video_path, fps, max_frames):
        """Extract frames from video at given FPS."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
        interval = max(1, round(src_fps / fps))
        frames = []
        idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % interval == 0:
                frames.append(frame)
                if max_frames and len(frames) >= max_frames:
                    break
            idx += 1

        cap.release()
        return frames


# ---------------------------------------------------------------------------
# Convert GCT output → Eridian PLY point cloud
# ---------------------------------------------------------------------------

def gct_to_pointcloud(result: dict, conf_threshold: float = 1.5,
                      subsample: int = 4):
    """Convert GCT predictions to (points_Nx3, colors_Nx3) for PLY export.

    Args:
        result: dict from GCTReconstructor with world_points, world_points_conf, images
        conf_threshold: minimum confidence to keep a point
        subsample: spatial subsampling factor (1 = all pixels, 4 = 1/16th)

    Returns:
        (points, colors) as numpy arrays, or (empty, empty) if no valid points.
    """
    world_pts = result["world_points"]      # [S, H, W, 3]
    world_conf = result["world_points_conf"]  # [S, H, W]
    images = result["images"]                # [S, 3, H, W] in [0, 1]

    S, H, W, _ = world_pts.shape

    all_pts = []
    all_cols = []

    for i in range(S):
        pts = world_pts[i]       # [H, W, 3]
        conf = world_conf[i]     # [H, W]
        img = images[i]          # [3, H, W]

        # Spatial subsample
        pts = pts[::subsample, ::subsample]    # [h, w, 3]
        conf = conf[::subsample, ::subsample]  # [h, w]
        img = img[:, ::subsample, ::subsample] # [3, h, w]

        # Confidence mask
        mask = conf > conf_threshold
        if not mask.any():
            continue

        pts_valid = pts[mask]                          # [N, 3]
        # Image is [3, h, w] → [h, w, 3] then mask
        colors_hw3 = (np.transpose(img, (1, 2, 0)) * 255).astype(np.uint8)
        cols_valid = colors_hw3[mask]                  # [N, 3] RGB

        all_pts.append(pts_valid)
        all_cols.append(cols_valid)

    if not all_pts:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

    points = np.concatenate(all_pts, axis=0).astype(np.float32)
    colors = np.concatenate(all_cols, axis=0).astype(np.uint8)
    return points, colors
