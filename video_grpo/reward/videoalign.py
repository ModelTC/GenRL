import torch
import numpy as np
import tempfile
import os
import sys
from typing import List, Union
from PIL import Image
import imageio
from loguru import logger

# Add VideoAlign to path for importing
_videoalign_path = os.path.join(os.path.dirname(__file__), "VideoAlign")
if _videoalign_path not in sys.path:
    sys.path.insert(0, _videoalign_path)

from inference import VideoVLMRewardInference
from .utils import prepare_images


# Global cache for VideoAlign inferencer
_inferencer_cache = {}


def _get_inferencer(
    checkpoint_path: str, device, dtype: torch.dtype
) -> VideoVLMRewardInference:
    """Get or create VideoAlign inferencer (cached per checkpoint/device/dtype).

    Args:
        checkpoint_path: Path to VideoAlign checkpoint directory.
        device: Device to run on (can be torch.device, str, or other device-like object).
        dtype: Data type for the model.

    Returns:
        VideoVLMRewardInference instance (cached per checkpoint/device/dtype).
    """
    # Normalize device to string for consistent caching and VideoVLMRewardInference compatibility
    if isinstance(device, torch.device):
        device_str = str(device)
    elif isinstance(device, str):
        device_str = device
    else:
        # Try to convert to string (handles cases like cuda:0, cpu, etc.)
        device_str = str(device)

    cache_key = (checkpoint_path, device_str, dtype)
    if cache_key not in _inferencer_cache:
        _inferencer_cache[cache_key] = VideoVLMRewardInference(
            load_from_pretrained=checkpoint_path,
            device=device_str,
            dtype=dtype,
        )
    return _inferencer_cache[cache_key]


def _convert_to_grayscale(images: np.ndarray) -> np.ndarray:
    """Convert RGB images/videos to grayscale.

    Args:
        images: Numpy array in NHWC (images) or NFHWC (videos) format (uint8).

    Returns:
        Grayscale images/videos in NHWC or NFHWC format (uint8).
    """
    if images.ndim == 4:  # NHWC (images)
        # Convert RGB to grayscale using standard weights: 0.299*R + 0.587*G + 0.114*B
        gray = np.dot(images[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
        # Expand to 3 channels: (N, H, W) -> (N, H, W, 3)
        gray = np.stack([gray, gray, gray], axis=-1)
        return gray
    elif images.ndim == 5:  # NFHWC (videos)
        # Convert RGB to grayscale: (N, F, H, W, C) -> (N, F, H, W)
        gray = np.dot(images[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
        # Expand to 3 channels: (N, F, H, W) -> (N, F, H, W, 3)
        gray = np.stack([gray, gray, gray], axis=-1)
        return gray
    else:
        raise ValueError(
            f"Unsupported array shape for grayscale conversion: {images.shape}"
        )


def _save_video_to_temp(frames: np.ndarray, fps: float = 8.0) -> str:
    """Save a video (numpy array of frames) to a temporary mp4 file.

    Args:
        frames: Frames as numpy array in FHWC format (uint8).
        fps: Frames per second for the output video.

    Returns:
        Path to the temporary mp4 file.
    """
    # Ensure frames are in FHWC format and uint8
    if frames.dtype != np.uint8:
        frames = frames.astype(np.uint8)

    temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    temp_file.close()

    # Save using imageio (similar to utils.py:log_videos)
    imageio.mimsave(temp_file.name, frames, fps=fps, codec="libx264", format="FFMPEG")

    return temp_file.name


def videoalign_mq_score(device, checkpoint_path: str = None):
    """
    VideoAlign-based reward for Motion Quality assessment.
    Uses grayscale videos to focus on motion characteristics rather than color.

    Args:
        device: Device to run the model on (can be torch.device, str, or other device-like object).
        checkpoint_path: Path to VideoAlign checkpoint directory.
                         Defaults to './video_grpo/reward/VideoAlign/checkpoints'.

    Returns:
        A function that takes (images, prompts, metadata, only_strict) and returns ({"avg": rewards}, {}).
    """
    if checkpoint_path is None:
        # Default to VideoAlign checkpoints directory
        checkpoint_path = os.path.join(
            os.path.dirname(__file__), "VideoAlign", "checkpoints"
        )

    # Resolve absolute path
    checkpoint_path = os.path.abspath(checkpoint_path)

    # Normalize device for tensor creation (torch.device or str both work)
    if isinstance(device, torch.device):
        device_for_tensor = device
    elif isinstance(device, str):
        device_for_tensor = torch.device(device)
    else:
        device_for_tensor = torch.device(str(device))

    dtype = torch.bfloat16  # VideoAlign typically uses bfloat16
    inferencer = _get_inferencer(checkpoint_path, device, dtype)

    def _fn(
        images: Union[List[Image.Image], np.ndarray, torch.Tensor],
        prompts: List[str],
        metadata=None,
        only_strict: bool = True,
    ):
        images_np, is_video = prepare_images(images)

        # Convert to grayscale for MQ assessment
        images_np = _convert_to_grayscale(images_np)

        rewards = []
        temp_files = []  # Track temporary files for cleanup

        try:
            batch_size = images_np.shape[0]

            for i in range(batch_size):
                if is_video:
                    # Video: process all frames (NFHWC format)
                    frames = images_np[i]  # Shape: (F, H, W, C)
                else:
                    # Image: treat as single-frame video
                    frames = images_np[i : i + 1]  # Shape: (1, H, W, C)

                # Save video to temporary file
                video_path = _save_video_to_temp(frames, fps=8.0)
                temp_files.append(video_path)

                # Get reward using VideoAlign
                # VideoAlign's reward method expects video_paths as list[str] and prompts as list[str]
                with torch.no_grad():
                    video_rewards = inferencer.reward(
                        video_paths=[video_path],
                        prompts=[prompts[i]],
                        use_norm=True,
                    )

                # Extract MQ score (Motion Quality)
                mq_score = video_rewards[0]["MQ"]
                rewards.append(mq_score)

            rewards = torch.tensor(
                rewards, dtype=torch.float32, device=device_for_tensor
            )
            return {"avg": rewards}, {}

        finally:
            # Clean up temporary files
            for temp_file in temp_files:
                try:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                except Exception as e:
                    logger.warning(f"Failed to delete temporary file {temp_file}: {e}")

    return _fn


def videoalign_ta_score(device, checkpoint_path: str = None):
    """
    VideoAlign-based reward for Text-Video Alignment assessment.
    Uses original color videos to preserve semantic correspondence assessment.

    Args:
        device: Device to run the model on (can be torch.device, str, or other device-like object).
        checkpoint_path: Path to VideoAlign checkpoint directory.
                         Defaults to './video_grpo/reward/VideoAlign/checkpoints'.

    Returns:
        A function that takes (images, prompts, metadata, only_strict) and returns ({"avg": rewards}, {}).
    """
    if checkpoint_path is None:
        # Default to VideoAlign checkpoints directory
        checkpoint_path = os.path.join(
            os.path.dirname(__file__), "VideoAlign", "checkpoints"
        )

    # Resolve absolute path
    checkpoint_path = os.path.abspath(checkpoint_path)

    # Normalize device for tensor creation (torch.device or str both work)
    if isinstance(device, torch.device):
        device_for_tensor = device
    elif isinstance(device, str):
        device_for_tensor = torch.device(device)
    else:
        device_for_tensor = torch.device(str(device))

    dtype = torch.bfloat16  # VideoAlign typically uses bfloat16
    inferencer = _get_inferencer(checkpoint_path, device, dtype)

    def _fn(
        images: Union[List[Image.Image], np.ndarray, torch.Tensor],
        prompts: List[str],
        metadata=None,
        only_strict: bool = True,
    ):
        images_np, is_video = prepare_images(images)
        # For TA, use original color (no grayscale conversion)

        rewards = []
        temp_files = []  # Track temporary files for cleanup

        try:
            batch_size = images_np.shape[0]

            for i in range(batch_size):
                if is_video:
                    # Video: process all frames (NFHWC format)
                    frames = images_np[i]  # Shape: (F, H, W, C)
                else:
                    # Image: treat as single-frame video
                    frames = images_np[i : i + 1]  # Shape: (1, H, W, C)

                # Save video to temporary file
                video_path = _save_video_to_temp(frames, fps=8.0)
                temp_files.append(video_path)

                # Get reward using VideoAlign
                # VideoAlign's reward method expects video_paths as list[str] and prompts as list[str]
                with torch.no_grad():
                    video_rewards = inferencer.reward(
                        video_paths=[video_path],
                        prompts=[prompts[i]],
                        use_norm=True,
                    )

                # Extract TA score (Text-Video Alignment)
                ta_score = video_rewards[0]["TA"]
                rewards.append(ta_score)

            rewards = torch.tensor(
                rewards, dtype=torch.float32, device=device_for_tensor
            )
            return {"avg": rewards}, {}

        finally:
            # Clean up temporary files
            for temp_file in temp_files:
                try:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                except Exception as e:
                    logger.warning(f"Failed to delete temporary file {temp_file}: {e}")

    return _fn
