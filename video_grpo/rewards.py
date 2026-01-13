import importlib
import torch
import inspect
from video_grpo.reward import (
    video_ocr_score,
    hpsv3_general_score,
    hpsv3_percentile_score,
    videoalign_mq_score,
    videoalign_ta_score,
)


def load_reward_fn(name: str, device, module_path: str | None = None):
    """Load a reward function by name, optionally from a user module."""
    if module_path:
        mod = importlib.import_module(module_path)
        fn = getattr(mod, f"{name}_score", None)
        if fn is None:
            raise ValueError(f"Reward {name}_score not found in {module_path}")
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
        accepts_device = any(
            p.name == "device" or p.name == "dev" for p in sig.parameters.values()
        )
        return fn(device) if accepts_device else fn()

    # Fallback: zero reward
    def _fn(images, prompts, metadata, only_strict=False):
        batch = len(prompts) if prompts is not None else 1
        zeros = torch.zeros(batch, device=device)
        return {"avg": zeros}, {}

    return _fn


def multi_score(device, reward_cfg, module_path: str | None = None):
    """Compose multiple reward heads defined in reward_cfg dict name->weight."""
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
            scores[name] = val * weights[name]
        stacked = torch.stack([v for v in scores.values()], dim=0)
        scores["avg"] = stacked.mean(0)
        return scores, {}

    return _fn
