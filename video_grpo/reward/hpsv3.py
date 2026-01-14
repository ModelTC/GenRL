import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import torch
from PIL import Image
from loguru import logger

# Prefer local HPSv3 submodule over site-packages.
_HPSV3_ROOT = Path(__file__).resolve().parent / "HPSv3"
if _HPSV3_ROOT.exists():
    hpsv3_path = str(_HPSV3_ROOT)
    if hpsv3_path not in sys.path:
        sys.path.insert(0, hpsv3_path)

from hpsv3 import HPSv3RewardInferencer

from .utils import prepare_images, preserve_accelerate_state
from video_grpo.utils import fast_init

# Global cache for HPSv3 inferencers to avoid loading the same model multiple times
_inferencer_cache: Dict[torch.device, HPSv3RewardInferencer] = {}


def _extract_reward_scalar(reward) -> float:
    if isinstance(reward, (list, tuple)) and len(reward) > 0:
        return float(reward[0])
    if isinstance(reward, torch.Tensor):
        if reward.numel() == 0:
            return 0.0
        return reward.flatten()[0].item()
    if hasattr(reward, "item"):
        return reward.item()
    return float(reward)


def _get_hpsv3_inferencer(device) -> HPSv3RewardInferencer:
    """Get or create HPSv3 inferencer for the given device.

    Args:
        device: Device to load the inferencer on (can be torch.device, str, or other device-like object).

    Returns:
        HPSv3RewardInferencer instance (shared across calls for the same device).
    """
    # Normalize device to torch.device for consistent caching
    if isinstance(device, torch.device):
        device_key = device
    elif isinstance(device, str):
        device_key = torch.device(device)
    else:
        # Try to convert to torch.device (handles cases like cuda:0, cpu, etc.)
        device_key = torch.device(device)

    # Check if we already have an inferencer for this device
    if device_key not in _inferencer_cache:
        # Use fast_init to avoid slow CPU initializations
        # Preserve Accelerate state in case HPSv3's TrainingArguments resets it.
        with preserve_accelerate_state():
            with fast_init(device_key, init_weights=False):
                _inferencer_cache[device_key] = HPSv3RewardInferencer(device=device_key)

    return _inferencer_cache[device_key]


def _save_frame_to_temp(frame: np.ndarray) -> str:
    """Save a frame (numpy array) to a temporary file.

    Args:
        frame: Frame as numpy array in HWC format (uint8).

    Returns:
        Path to the temporary file.
    """
    pil_image = Image.fromarray(frame)
    temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    temp_file.close()
    pil_image.save(temp_file.name)
    return temp_file.name


def hpsv3_general_score(device):
    """
    HPSv3-based reward for visual quality assessment.
    Uses the general prompt "A high-quality image" for all frames.
    Returns the mean score of all frames.

    Returns:
        A function that takes (images, prompts, metadata, only_strict) and returns ({"avg": rewards}, {}).
    """
    inferencer = _get_hpsv3_inferencer(device)
    device_type = torch.device(device).type if not isinstance(device, torch.device) else device.type
    general_prompt = "A high-quality image"

    def _fn(
        images: Union[List[Image.Image], np.ndarray, torch.Tensor],
        prompts: List[str],
        metadata=None,
        only_strict: bool = True,
    ):
        images_np, is_video = prepare_images(images)
        rewards = []
        temp_files = []  # Track temporary files for cleanup

        try:
            # Handle batch dimension
            if is_video:
                # images_np shape: (N, F, H, W, C)
                batch_size = images_np.shape[0]
            else:
                # images_np shape: (N, H, W, C)
                batch_size = images_np.shape[0]

            for i in range(batch_size):
                frame_rewards = []
                frame_paths = []

                if is_video:
                    # Video: process all frames
                    frames = images_np[i]  # Shape: (F, H, W, C)
                else:
                    # Image: treat as single frame
                    frames = [images_np[i]]  # Shape: (H, W, C)

                # Save frames to temporary files
                for frame in frames:
                    frame_path = _save_frame_to_temp(frame)
                    frame_paths.append(frame_path)
                    temp_files.append(frame_path)

                # Evaluate all frames with the general prompt
                # Repeat the general prompt for all frames
                frame_prompts = [general_prompt] * len(frame_paths)
                with torch.no_grad():
                    with torch.amp.autocast(device_type=device_type):
                        frame_rewards_raw = inferencer.reward(
                            frame_prompts, image_paths=frame_paths
                        )

                # Extract mu values (mean scores)
                # HPSv3 returns a list where each element is [mu, sigma] or a tensor
                for reward in frame_rewards_raw:
                    score = _extract_reward_scalar(reward)
                    frame_rewards.append(score)

                # Calculate mean score across all frames
                video_reward = np.mean(frame_rewards) if frame_rewards else 0.0
                rewards.append(video_reward)

            rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
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


def hpsv3_percentile_score(device):
    """
    HPSv3-based reward for text-video alignment assessment.
    Uses the video caption (prompts) for evaluation.
    Returns the mean score of the top 30% frames.

    Returns:
        A function that takes (images, prompts, metadata, only_strict) and returns ({"avg": rewards}, {}).
    """
    inferencer = _get_hpsv3_inferencer(device)
    device_type = torch.device(device).type if not isinstance(device, torch.device) else device.type

    def _fn(
        images: Union[List[Image.Image], np.ndarray, torch.Tensor],
        prompts: List[str],
        metadata=None,
        only_strict: bool = True,
    ):
        images_np, is_video = prepare_images(images)
        rewards = []
        temp_files = []  # Track temporary files for cleanup

        try:
            # Handle batch dimension
            if is_video:
                # images_np shape: (N, F, H, W, C)
                batch_size = images_np.shape[0]
            else:
                # images_np shape: (N, H, W, C)
                batch_size = images_np.shape[0]

            for i, prompt in enumerate(prompts):
                frame_rewards = []
                frame_paths = []

                if is_video:
                    # Video: process all frames
                    frames = images_np[i]  # Shape: (F, H, W, C)
                else:
                    # Image: treat as single frame
                    frames = [images_np[i]]  # Shape: (H, W, C)

                # Save frames to temporary files
                for frame in frames:
                    frame_path = _save_frame_to_temp(frame)
                    frame_paths.append(frame_path)
                    temp_files.append(frame_path)

                # Evaluate all frames with the video caption (prompt)
                # Use the same prompt for all frames in the video
                frame_prompts = [prompt] * len(frame_paths)
                with torch.no_grad():
                    with torch.amp.autocast(device_type=device_type):
                        frame_rewards_raw = inferencer.reward(
                            frame_prompts, image_paths=frame_paths
                        )

                # Extract mu values (mean scores)
                # HPSv3 returns a list where each element is [mu, sigma] or a tensor
                for reward in frame_rewards_raw:
                    score = _extract_reward_scalar(reward)
                    frame_rewards.append(score)

                # Calculate mean score of top 30% frames
                if frame_rewards:
                    frame_rewards_sorted = sorted(frame_rewards, reverse=True)
                    top_30_percent_count = max(1, int(len(frame_rewards_sorted) * 0.3))
                    video_reward = np.mean(frame_rewards_sorted[:top_30_percent_count])
                else:
                    video_reward = 0.0

                rewards.append(video_reward)

            rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
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
