import os
import random
import json
import hashlib
import contextlib
from typing import Any, Dict, List, Optional, Tuple, Type

import imageio
import wandb
import numpy as np
from accelerate import Accelerator, FullyShardedDataParallelPlugin, ProjectConfiguration
from diffusers.utils.torch_utils import is_compiled_module
import torch
from torch.nn.utils import no_init_weights

# fast init helpers
_ORIGINAL_INITS: Dict[Type[torch.nn.Module], Any] = {
    torch.nn.Linear: torch.nn.Linear.__init__,
    torch.nn.Embedding: torch.nn.Embedding.__init__,
    torch.nn.LayerNorm: torch.nn.LayerNorm.__init__,
}


def _get_fast_init(cls: Type[torch.nn.Module], device: torch.device):
    assert cls in _ORIGINAL_INITS

    def _fast_init(self, *args, **kwargs):
        # Same as torch.nn.utils.skip_init, excluding checks
        if "device" in kwargs:
            kwargs.pop("device")
        _ORIGINAL_INITS[cls](self, *args, **kwargs, device="meta")
        self.to_empty(device=device)

    return _fast_init


@contextlib.contextmanager
def fast_init(device: torch.device, init_weights: bool = False):
    """
    Avoid multiple slow CPU initializations by constructing modules on meta device,
    then materializing on the target device.
    """
    for cls in _ORIGINAL_INITS:
        cls.__init__ = _get_fast_init(cls, device)

    with contextlib.nullcontext() if init_weights else no_init_weights():
        yield

    for cls in _ORIGINAL_INITS:
        cls.__init__ = _ORIGINAL_INITS[cls]


def build_accelerator(
    cfg: Any, grad_acc_steps: int, project_config: ProjectConfiguration
) -> Accelerator:
    """Construct an Accelerate `Accelerator` with FSDP plugin.

    Args:
        cfg: Parsed training config containing accelerate settings.
        grad_acc_steps: Gradient accumulation steps for Accelerator.
        project_config: Accelerate `ProjectConfiguration`.

    Returns:
        Configured `Accelerator` instance.
    """
    fsdp_cfg = cfg.accelerate.fsdp_config.__dict__
    fsdp_plugin = FullyShardedDataParallelPlugin(**fsdp_cfg)
    accelerator = Accelerator(
        log_with="wandb",
        mixed_precision=cfg.accelerate.mixed_precision,
        project_config=project_config,
        gradient_accumulation_steps=grad_acc_steps,
        fsdp_plugin=fsdp_plugin,
    )
    return accelerator


def unwrap_model(model: torch.nn.Module, accelerator: Accelerator) -> torch.nn.Module:
    """Remove accelerator/compile wrappers to access the base model.

    Args:
        model: Model possibly wrapped by Accelerate/compile.
        accelerator: Accelerator used for training.

    Returns:
        Unwrapped base model.
    """
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model


def resolve_resume_checkpoint(resume_from: Optional[str]) -> Optional[str]:
    """Resolve explicit checkpoint or latest checkpoint-* within a directory.

    Args:
        resume_from: Optional path to a checkpoint dir or parent directory.

    Returns:
        Path to a concrete `checkpoint-*` directory or None if not found.
    """
    if not resume_from:
        return None
    resume_from = os.path.normpath(os.path.expanduser(resume_from))
    if "checkpoint-" in os.path.basename(resume_from):
        return resume_from
    if not os.path.isdir(resume_from):
        return None
    checkpoints = [c for c in os.listdir(resume_from) if "checkpoint-" in c]
    if not checkpoints:
        return None
    latest = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))[-1]
    return os.path.join(resume_from, latest)


def create_generator(
    prompts: List[str], base_seed: int, device: Optional[torch.device] = None
) -> List[torch.Generator]:
    """Create a deterministic torch.Generator per prompt, seeded by prompt hash + base_seed.

    Args:
        prompts: List of prompt strings.
        base_seed: Base seed for reproducibility.
        device: Optional device for generators (defaults to CUDA if available).

    Returns:
        List of torch.Generator objects (one per prompt).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generators: List[torch.Generator] = []
    for prompt in prompts:
        hash_digest = hashlib.sha256(prompt.encode()).digest()
        prompt_hash_int = int.from_bytes(hash_digest[:4], "big")
        seed = (base_seed + prompt_hash_int) % (2**31)
        gen = torch.Generator(device=device).manual_seed(seed)
        generators.append(gen)
    return generators


def log_videos(
    tag: str,
    cfg: Any,
    accelerator: Accelerator,
    videos: torch.Tensor,
    prompts: List[str],
    rewards: Dict[str, Any],
    step: int,
) -> None:
    """Log sampled videos to disk and wandb.

    Args:
        tag: Label to distinguish train/eval logs.
        cfg: Training config for paths/run_name.
        accelerator: Accelerator for process control/logging.
        videos: Tensor of shape (B, T, C, H, W).
        prompts: List of prompt strings.
        rewards: Dict containing at least key 'avg'.
        step: Global step for logging.
    """
    video_dir = os.path.join(cfg.paths.logdir, cfg.run_name, f"{tag}_videos")
    os.makedirs(video_dir, exist_ok=True)
    num_samples = min(15, len(videos))
    sample_indices = random.sample(range(len(videos)), num_samples)
    video_paths = []
    for idx, i in enumerate(sample_indices):
        video = videos[i]
        frames = [img for img in video.cpu().numpy().transpose(0, 2, 3, 1)]
        frames = [(frame * 255).astype(np.uint8) for frame in frames]
        out_path = os.path.join(video_dir, f"{tag}_{step}_{idx}.mp4")
        imageio.mimsave(out_path, frames, fps=8, codec="libx264", format="FFMPEG")
        video_paths.append(out_path)
    accelerator.log(
        {
            f"{tag}_video": [
                wandb.Video(
                    path,
                    caption=f"{prompt:.100} | avg: {avg_reward:.2f}",
                    format="mp4",
                    fps=8,
                )
                for path, prompt, avg_reward in zip(
                    video_paths,
                    [prompts[i] for i in sample_indices],
                    [rewards["avg"][i] for i in sample_indices],
                )
            ]
        },
        step=step,
    )


def save_ckpt(
    cfg: Any,
    transformer: torch.nn.Module,
    pipeline: Any,
    global_step: int,
    epoch: int,
    accelerator: Accelerator,
    ema: Optional[Any],
    transformer_params: List[torch.nn.Parameter],
    current_epoch_tag: int,
    full_finetune: bool,
) -> None:
    """Save state, EMA, metadata, and unwrapped model checkpoint.

    Args:
        cfg: Training config.
        transformer: Potentially wrapped transformer being trained.
        pipeline: Diffusion pipeline (for saving unwrapped).
        global_step: Current global step.
        epoch: Current epoch.
        accelerator: Accelerator for saving state.
        ema: EMA wrapper (may be None if disabled).
        transformer_params: List of trainable parameters for EMA swap.
        current_epoch_tag: Sampler epoch_tag for resume alignment.
        full_finetune: Whether running full-parameter finetune.
    """
    save_root = os.path.join(
        cfg.paths.save_dir, "checkpoints", f"checkpoint-{global_step}"
    )
    os.makedirs(save_root, exist_ok=True)
    metadata = {
        "global_step": global_step,
        "epoch": epoch,
        "current_epoch_tag": current_epoch_tag,
        "run_name": cfg.run_name,
    }
    accelerator.save_state(save_root)
    if accelerator.is_main_process:
        if cfg.train.ema:
            torch.save(ema.state_dict(), os.path.join(save_root, "ema_state.pt"))

        unwrap_dir = os.path.join(save_root, "unwrapped_model")
        os.makedirs(unwrap_dir, exist_ok=True)

        if cfg.train.ema:
            ema.copy_ema_to(transformer_params, store_temp=True)
        base_transformer = unwrap_model(transformer, accelerator)
        transformer_dir = os.path.join(unwrap_dir, "transformer")
        base_transformer.save_pretrained(transformer_dir)
        with open(os.path.join(save_root, "metadata.json"), "w") as f:
            json.dump(metadata, f)
        if cfg.train.ema:
            ema.copy_temp_to(transformer_params)


def calculate_zero_std_ratio(
    prompts: List[str], gathered_rewards: Dict[str, np.ndarray]
) -> float:
    """Compute zero-std ratio for rewards grouped by prompt.

    Args:
        prompts: List of prompt strings.
        gathered_rewards: Dict containing 'ori_avg' reward array.

    Returns:
        Fraction of prompts whose reward std is exactly zero.
    """
    prompt_array = np.array(prompts)
    unique_prompts, inverse_indices, counts = np.unique(
        prompt_array, return_inverse=True, return_counts=True
    )
    grouped_rewards = gathered_rewards["ori_avg"][np.argsort(inverse_indices)]
    split_indices = np.cumsum(counts)[:-1]
    reward_groups = np.split(grouped_rewards, split_indices)
    prompt_std_devs = np.array([np.std(group) for group in reward_groups])
    zero_std_count = np.count_nonzero(prompt_std_devs == 0)
    zero_std_ratio = (
        zero_std_count / len(prompt_std_devs) if len(prompt_std_devs) else 0.0
    )
    return zero_std_ratio
