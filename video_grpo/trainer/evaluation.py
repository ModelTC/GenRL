"""Evaluation utilities for trainers."""

from collections import defaultdict
from typing import Any, Callable, List, Optional
import numpy as np
import torch
import tqdm
from accelerate import Accelerator
from accelerate.utils import set_seed
from loguru import logger

from video_grpo.config import Config
from video_grpo.trainer.embeddings import wan_compute_text_embeddings
from video_grpo.diffusers_patch.wan_pipeline_with_logprob import (
    wan_pipeline_with_logprob,
)
from video_grpo.utils import log_videos

tqdm = tqdm.tqdm


def wan_eval_once(
    cfg: Config,
    accelerator: Accelerator,
    pipeline: Any,  # Pipeline with scheduler and tokenizer
    test_dataloader: torch.utils.data.DataLoader,
    text_encoders: List[Any],
    tokenizers: List[Any],
    sample_neg_prompt_embeds: torch.Tensor,
    eval_reward_fn: Callable,
    autocast: Any,
    global_step: int,
    ema: Any | None,
    transformer_params: List[torch.nn.Parameter] | None,
    log_metrics: Optional[Callable[[Accelerator, dict, int], None]] = None,
):
    """Full eval loop aligned to original behavior: iterate all batches, log rewards/videos.

    Args:
        cfg: Training config.
        accelerator: Accelerator handle.
        pipeline: Diffusion pipeline.
        test_dataloader: Evaluation dataloader.
        text_encoders: List of text encoders.
        tokenizers: List of tokenizers.
        sample_neg_prompt_embeds: Negative embeddings for CFG.
        eval_reward_fn: Reward function for eval.
        autocast: Autocast context manager.
        global_step: Current global step for logging.
        ema: EMA module wrapper.
        transformer_params: List of transformer parameters.
        log_metrics: Optional logging helper. If provided, used instead of
            calling `accelerator.log` directly for scalar metrics.

    Returns:
        None. Logs metrics/videos to accelerator.
    """
    set_seed(cfg.seed, device_specific=True)
    if cfg.train.ema and ema is not None and transformer_params is not None:
        ema.copy_ema_to(transformer_params, store_temp=True)
    all_rewards = defaultdict(list)
    last_batch = None
    eval_guidance = (
        getattr(cfg.sample, "eval_guidance_scale", None) or cfg.sample.guidance_scale
    )
    for test_batch in tqdm(
        test_dataloader,
        desc="Eval",
        disable=not accelerator.is_local_main_process,
        position=0,
    ):
        _, test_prompts, test_metadata = test_batch
        test_embeds = wan_compute_text_embeddings(
            test_prompts,
            text_encoders,
            tokenizers,
            max_sequence_length=512,
            device=accelerator.device,
        )
        with autocast():
            with torch.no_grad():
                videos_eval, _, _, _, _ = wan_pipeline_with_logprob(
                    pipeline,
                    prompt_embeds=test_embeds,
                    negative_prompt_embeds=sample_neg_prompt_embeds[: len(test_embeds)],
                    num_inference_steps=cfg.sample.eval_num_steps,
                    guidance_scale=eval_guidance,
                    output_type="pt",
                    return_dict=False,
                    num_frames=cfg.frames,
                    height=cfg.height,
                    width=cfg.width,
                    deterministic=True,
                    kl_reward=0,
                    noise_level=cfg.sample.noise_level,
                    sde_type=cfg.sample.sde_type,
                    diffusion_clip=cfg.sample.diffusion_clip,
                    diffusion_clip_value=cfg.sample.diffusion_clip_value,
                    sde_window_size=0,
                    sde_window_range=None,
                    # For evaluation, we don't need to compute KL reward and
                    # don't need sde_window_size and sde_window_range
                    # because we are not training
                )
        rewards_eval, reward_meta = eval_reward_fn(
            videos_eval, test_prompts, test_metadata, False
        )
        last_batch = (videos_eval, test_prompts, rewards_eval, reward_meta)
        for key, value in rewards_eval.items():
            gathered = (
                accelerator.gather(torch.as_tensor(value, device=accelerator.device))
                .cpu()
                .numpy()
            )
            all_rewards[key].append(gathered)

    # concat and log
    all_rewards = {k: np.concatenate(v) for k, v in all_rewards.items()}
    if accelerator.is_main_process:
        # Only log raw scores (ending with '_raw')
        raw_keys = [k for k in all_rewards.keys() if k.endswith("_raw")]
        metrics = {
            f"eval_reward_{k}": float(np.mean(v))
            for k, v in all_rewards.items()
            if k in raw_keys
        }
        if log_metrics is not None:
            # Use the shared logging helper (will also print to stdout)
            log_metrics(accelerator, metrics, global_step)
        else:
            accelerator.log(metrics, step=global_step)

        logger.info(f"Eval rewards: {all_rewards}")
        videos_eval, test_prompts, rewards_eval, _ = last_batch
        log_videos(
            "eval",
            cfg,
            accelerator,
            videos_eval,
            test_prompts,
            rewards_eval,
            global_step,
        )
    # restore weights after eval
    if cfg.train.ema and ema is not None and transformer_params is not None:
        ema.copy_temp_to(transformer_params)
