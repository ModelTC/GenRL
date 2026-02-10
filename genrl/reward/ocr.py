
import numpy as np
import torch
from Levenshtein import distance
from paddleocr import PaddleOCR
from PIL import Image

from .utils import prepare_images


def video_ocr_score():
    """
    OCR-based reward for images or videos. Videos are sampled every few frames.
    Returns a dict with key 'avg'.
    """
    # Must run on CPU.
    ocr = PaddleOCR(use_angle_cls=False, lang="en", use_gpu=False, show_log=False)
    frame_interval = 4

    def _fn(
        images: list[Image.Image] | np.ndarray,
        prompts: list[str],
        metadata=None,
        only_strict: bool = True,
    ):
        # Align with flow_grpo: take text inside quotes if present, strip spaces/lower.
        prompts_clean = [p.split('"')[1] if '"' in p else p for p in prompts]
        prompts_clean = [p.replace(" ", "").lower() for p in prompts_clean]
        images_np, _ = prepare_images(images)

        rewards = []
        for img, prompt in zip(images_np, prompts_clean, strict=False):
            frames = img[::frame_interval] if img.ndim == 4 else [img]
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

        rewards = torch.tensor(rewards, dtype=torch.float32)
        return {"avg": rewards}, {}

    return _fn
