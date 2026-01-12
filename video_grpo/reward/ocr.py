import torch
import numpy as np
from typing import List, Union
from PIL import Image
from paddleocr import PaddleOCR
from Levenshtein import distance


def _prepare_images(images):
    if isinstance(images, torch.Tensor):
        # Align with flow_grpo: handle 4D (NCHW) and 5D (NFCHW) tensors.
        if images.dim() == 4 and images.shape[1] == 3:
            images = images.permute(0, 2, 3, 1)  # NCHW -> NHWC
        elif images.dim() == 5 and images.shape[2] == 3:
            images = images.permute(0, 1, 3, 4, 2)  # NFCHW -> NFHWC
        images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
    return images


def video_ocr_score(device):
    """
    OCR-based reward for images or videos. Videos are sampled every few frames.
    Returns a dict with key 'avg'.
    """
    ocr = PaddleOCR(
        use_angle_cls=False, lang="en", use_gpu=(device.type == "cuda"), show_log=False
    )
    frame_interval = 4

    def _fn(
        images: Union[List[Image.Image], np.ndarray],
        prompts: List[str],
        metadata=None,
        only_strict: bool = True,
    ):
        # Align with flow_grpo: take text inside quotes if present, strip spaces/lower.
        prompts_clean = [p.split('"')[1] if '"' in p else p for p in prompts]
        prompts_clean = [p.replace(" ", "").lower() for p in prompts_clean]
        images_np = _prepare_images(images)

        rewards = []
        for img, prompt in zip(images_np, prompts_clean):
            if img.ndim == 4:
                frames = img[::frame_interval]
            else:
                frames = [img]
            frame_rewards = []
            for frame in frames:
                try:
                    result = ocr.ocr(frame, cls=False)
                    text = (
                        "".join(
                            [res[1][0] if res[1][1] > 0 else "" for res in result[0]]
                        )
                        if result[0]
                        else ""
                    )
                    text = text.replace(" ", "").lower()
                    dist = distance(text, prompt)
                    dist = min(dist, len(prompt))
                except Exception:
                    dist = len(prompt)
                reward = 1 - dist / max(len(prompt), 1)
                # Keep positive frames (flow_grpo behavior), drop zero-only sequences.
                if reward > 0:
                    frame_rewards.append(reward)
            rewards.append(np.mean(frame_rewards) if frame_rewards else 0.0)

        rewards = torch.tensor(rewards, device=device, dtype=torch.float32)
        return {"avg": rewards}, {}

    return _fn
