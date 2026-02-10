import importlib
import inspect
from contextlib import contextmanager

import torch

from genrl.reward import (
    hpsv3_general_score,
    hpsv3_percentile_score,
    video_ocr_score,
    videoalign_mq_score,
    videoalign_ta_score,
)
from genrl.reward.hpsv3 import set_hpsv3_device
from genrl.reward.videoalign import set_videoalign_device
from genrl.utils import cleanup_memory


def load_reward_fn(name: str, device, module_path: str | None = None):
    """Load a reward function by name, optionally from a user module."""
    if module_path:
        mod = importlib.import_module(module_path)
        fn = getattr(mod, f"{name}_score", None)
        if fn is None:
            msg = f"Reward {name}_score not found in {module_path}"
            raise ValueError(msg)
        return fn(device) if callable(fn) else fn
    builtin = {
        "video_ocr": video_ocr_score,
        "hpsv3_general": hpsv3_general_score,
        "hpsv3_percentile": hpsv3_percentile_score,
        "videoalign_mq": videoalign_mq_score,
        "videoalign_ta": videoalign_ta_score,
    }
    if name in builtin:
        fn = builtin[name]
        sig = inspect.signature(fn)
        accepts_device = any(p.name in {"device", "dev"} for p in sig.parameters.values())
        return fn(device) if accepts_device else fn()

    # Fallback: zero reward
    def _fn(images, prompts, metadata, only_strict=False):
        batch = len(prompts) if prompts is not None else 1
        zeros = torch.zeros(batch, device=device)
        return {"avg": zeros}, {}

    return _fn


def multi_score(device, reward_cfg, module_path: str | None = None, return_raw_scores: bool = False):
    """Compose multiple reward heads defined in reward_cfg dict name->weight.

    Args:
        device: Device to run rewards on.
        reward_cfg: Dict mapping reward name to weight.
        module_path: Optional custom module path for reward functions.
        return_raw_scores: If True, also return unweighted scores in scores dict with '_raw' suffix.

    Returns:
        A function that returns (scores_dict, metadata_dict).
        If return_raw_scores=True, scores_dict contains both weighted (name) and raw (name_raw) scores.
    """
    reward_fns = {}
    weights = {}
    for name, weight in reward_cfg.items():
        reward_fns[name] = load_reward_fn(name, device, module_path)
        weights[name] = weight

    def _fn(images, prompts, metadata, only_strict=True):
        scores = {}
        for name, fn in reward_fns.items():
            out, _meta = fn(images, prompts, metadata)
            if isinstance(out, dict):
                # expect 'avg' key
                val = out.get("avg", out.get("reward", out))
            else:
                val = out
            if return_raw_scores:
                scores[f"{name}_raw"] = val  # Store raw (unweighted) scores
            scores[name] = val * weights[name]  # Store weighted scores
        stacked = torch.stack([scores[name] for name in reward_cfg], dim=0)
        scores["avg"] = stacked.mean(0)
        return scores, {}

    return _fn


_GPU_REWARD_NAMES = {
    "hpsv3_general",
    "hpsv3_percentile",
    "videoalign_mq",
    "videoalign_ta",
}


def _has_reward(reward_cfg, names) -> bool:
    if not reward_cfg:
        return False
    return any(name in reward_cfg for name in names)


def _device_type(device) -> str:
    if isinstance(device, torch.device):
        return device.type
    return torch.device(device).type


def move_reward_models(reward_cfg, device) -> None:
    """Move GPU-backed reward models to the given device."""
    if _has_reward(reward_cfg, {"hpsv3_general", "hpsv3_percentile"}):
        set_hpsv3_device(device)
    if _has_reward(reward_cfg, {"videoalign_mq", "videoalign_ta"}):
        set_videoalign_device(device)


@contextmanager
def reward_models_on_device(reward_cfg, device):
    """Temporarily move reward models to device, then back to CPU."""
    if _has_reward(reward_cfg, _GPU_REWARD_NAMES):
        use_cuda = _device_type(device) == "cuda"
        move_reward_models(reward_cfg, device)
        try:
            yield
        finally:
            move_reward_models(reward_cfg, "cpu")
            if use_cuda:
                cleanup_memory()
    else:
        yield
