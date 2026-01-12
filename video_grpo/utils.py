import os
import random
import json
import hashlib
import contextlib
import re
import shutil
from typing import Any, Dict, List, Optional, Tuple, Type

import imageio
import wandb
import numpy as np
from accelerate import Accelerator, FullyShardedDataParallelPlugin
from accelerate.utils import ProjectConfiguration
from diffusers.utils.torch_utils import is_compiled_module
import torch
from transformers.modeling_utils import no_init_weights

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
    # Use accelerator.unwrap_model which calls extract_model_from_parallel
    # Note: extract_model_from_parallel has recursive=False by default, which means
    # it only removes top-level FSDP wrapper, not nested ones. However, for PEFT models,
    # the FSDP wrapper should be at the top level, so this should be sufficient.
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
    # save_dir already contains run_name, so don't add it again
    video_dir = os.path.join(cfg.paths.save_dir, f"{tag}_videos")
    os.makedirs(video_dir, exist_ok=True)
    num_samples = min(15, len(videos))
    sample_indices = random.sample(range(len(videos)), num_samples)
    video_paths = []
    for idx, i in enumerate(sample_indices):
        video = videos[i]
        frames = [img for img in video.cpu().numpy().transpose(0, 2, 3, 1)]
        frames = [(frame * 255).astype(np.uint8) for frame in frames]
        out_path = os.path.join(video_dir, f"{tag}_{step}_{idx}.mp4")
        imageio.mimsave(out_path, frames, fps=16, codec="libx264", format="FFMPEG")
        video_paths.append(out_path)
    accelerator.log(
        {
            f"{tag}_video": [
                wandb.Video(
                    path,
                    caption=f"{prompt:.100} | avg: {avg_reward:.2f}",
                    format="mp4",
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
    global_step: int,
    epoch: int,
    accelerator: Accelerator,
    ema: Optional[Any],
    transformer_params: List[torch.nn.Parameter],
    current_epoch_tag: int,
) -> None:
    """Save state, EMA, metadata, and unwrapped model checkpoint.

    Args:
        cfg: Training config.
        transformer: Potentially wrapped transformer being trained.
        global_step: Current global step.
        epoch: Current epoch.
        accelerator: Accelerator for saving state.
        ema: EMA wrapper (may be None if disabled).
        transformer_params: List of trainable parameters for EMA swap.
        current_epoch_tag: Sampler epoch_tag for resume alignment.
    """
    # save_dir already contains run_name, so checkpoints will be saved in run directory
    checkpoints_dir = os.path.join(cfg.paths.save_dir, "checkpoints")
    save_root = os.path.join(checkpoints_dir, f"checkpoint-{global_step}")

    # Clean up old checkpoints if limit is set (only on main process)
    if (
        accelerator.is_main_process
        and cfg.num_checkpoint_limit is not None
        and cfg.num_checkpoint_limit > 0
    ):
        if os.path.exists(checkpoints_dir):
            # Get all checkpoint directories
            checkpoint_folders = []
            for item in os.listdir(checkpoints_dir):
                checkpoint_path = os.path.join(checkpoints_dir, item)
                if os.path.isdir(checkpoint_path) and item.startswith("checkpoint-"):
                    # Extract step number from folder name (e.g., "checkpoint-120" -> 120)
                    match = re.search(r"checkpoint-(\d+)", item)
                    if match:
                        step_num = int(match.group(1))
                        checkpoint_folders.append((step_num, checkpoint_path))

            # Sort by step number (oldest first)
            checkpoint_folders.sort(key=lambda x: x[0])

            # Delete oldest checkpoints if we exceed the limit
            # We check (len + 1) because we're about to save a new checkpoint
            if len(checkpoint_folders) + 1 > cfg.num_checkpoint_limit:
                num_to_delete = len(checkpoint_folders) + 1 - cfg.num_checkpoint_limit
                for step_num, folder_path in checkpoint_folders[:num_to_delete]:
                    shutil.rmtree(folder_path)
                    print(f"Deleted old checkpoint: checkpoint-{step_num}")

    if accelerator.is_main_process:
        os.makedirs(save_root, exist_ok=True)
    metadata = {
        "global_step": global_step,
        "epoch": epoch,
        "current_epoch_tag": current_epoch_tag,
        "run_name": cfg.run_name,
    }
    accelerator.save_state(save_root)
    # Save EMA per-rank to avoid FSDP shard mismatches.
    if cfg.train.ema:
        ema_dir = os.path.join(save_root, "ema")
        if accelerator.is_main_process:
            os.makedirs(ema_dir, exist_ok=True)
        accelerator.wait_for_everyone()
        ema_path = os.path.join(
            ema_dir, f"ema_state_rank{accelerator.process_index}.pt"
        )
        torch.save(ema.state_dict(), ema_path)

    # Apply EMA weights to model before saving (on all processes, before unwrap)
    if cfg.train.ema:
        ema.copy_ema_to(transformer_params, store_temp=True)

    # Unwrap model on all processes (needed to avoid FSDP unshard in save_pretrained)
    #
    # Evidence that save_pretrained triggers unshard without unwrap:
    # 1. FSDP.state_dict() calls _unshard_params_for_summon() to gather sharded parameters
    # 2. PEFT's save_pretrained() calls get_peft_model_state_dict() which calls model.state_dict()
    # 3. Without unwrap, model.state_dict() on FSDP-wrapped model triggers unshard operations
    # 4. Unshard requires all processes to participate via all_gather, which can deadlock
    #    if processes are not synchronized (e.g., only main process calls save_pretrained)
    #
    # Solution: unwrap_model() removes FSDP wrapper via extract_model_from_parallel(),
    # which checks isinstance(model, FSDP) and extracts the underlying module.
    # After unwrap, state_dict() operates on the base model without triggering unshard.
    base_transformer = unwrap_model(transformer, accelerator)

    # Prepare directories (only on main process)
    if accelerator.is_main_process:
        unwrap_dir = os.path.join(save_root, "unwrapped_model")
        os.makedirs(unwrap_dir, exist_ok=True)
        transformer_dir = os.path.join(unwrap_dir, "transformer")
    else:
        transformer_dir = None

    # Critical: save_pretrained may call model.state_dict() internally:
    # - For LoRA: PeftModel.save_pretrained -> get_peft_model_state_dict -> model.state_dict()
    # - For full finetune: PreTrainedModel.save_pretrained -> model.state_dict()
    # Even after unwrap, if there are nested FSDP modules, state_dict() may trigger unshard
    # Solution: Get state_dict on ALL processes first (this ensures all processes participate in unshard if needed)
    # Then pass it to save_pretrained to avoid calling model.state_dict() again
    # This is safer than relying on unwrap to remove all FSDP wrappers
    # Note: Even if unwrap removed top-level FSDP, nested FSDP modules may still exist.
    # Calling state_dict() on all processes ensures all participate in unshard if needed.
    # If unshard is triggered, it will synchronize all processes internally via all_gather
    state_dict_to_save = base_transformer.state_dict()

    try:
        if accelerator.is_main_process:
            # Pass state_dict explicitly to avoid save_pretrained calling model.state_dict() again
            base_transformer.save_pretrained(
                transformer_dir, state_dict=state_dict_to_save
            )

            with open(os.path.join(save_root, "metadata.json"), "w") as f:
                json.dump(metadata, f)
    finally:
        if cfg.train.ema:
            ema.copy_temp_to(transformer_params)

        accelerator.wait_for_everyone()


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
