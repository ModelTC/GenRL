"""Sampling utilities for trainers."""
import time
from typing import Any, Callable, Dict, List
import torch
import tqdm
from accelerate import Accelerator
from loguru import logger

from video_grpo.config import Config
from video_grpo.trainer.embeddings import wan_compute_text_embeddings
from video_grpo.diffusers_patch.wan_pipeline_with_logprob import (
    wan_pipeline_with_logprob,
)
from video_grpo.utils import create_generator, log_videos

tqdm = tqdm.tqdm


def wan_sample_epoch(
    cfg: Config,
    accelerator: Accelerator,
    pipeline: Any,  # Pipeline with scheduler and tokenizer
    train_sampler: Any,
    train_iter,
    reward_fn: Callable,
    sample_neg_prompt_embeds: torch.Tensor,
    text_encoders: List[Any],
    tokenizers: List[Any],
    executor: Any,  # ThreadPoolExecutor
    autocast: Any,
    epoch: int,
    global_step: int,
) -> List[Dict[str, Any]]:
    """Sampling epoch that returns gathered samples and logs train videos.

    Args:
        cfg: Training config.
        accelerator: Accelerator handle.
        pipeline: Diffusion pipeline.
        train_sampler: Sampler that tags batches with epoch.
        train_iter: Infinite dataloader iterator.
        reward_fn: Reward function for training.
        sample_neg_prompt_embeds: Negative embeddings for CFG.
        text_encoders: List of text encoders.
        tokenizers: List of tokenizers.
        executor: ThreadPoolExecutor for async reward compute.
        autocast: Autocast context manager.
        epoch: Current outer epoch id.
        global_step: Global step for logging.

    Returns:
        List of sample dicts ready for collation.
    """
    samples = []
    train_video_cache = []
    for i in tqdm(
        range(cfg.sample.num_batches_per_epoch),
        desc=f"Epoch {epoch}: sampling",
        disable=not accelerator.is_local_main_process,
        position=0,
    ):
        current_epoch_tag = epoch * cfg.sample.num_batches_per_epoch + i
        train_sampler.set_epoch(current_epoch_tag)

        # Drain prefetched batches until sampler epoch matches to avoid cross-epoch mixing.
        while True:
            epoch_tag, prompts, prompt_metadata = next(train_iter)
            if epoch_tag == current_epoch_tag:
                break

        prompt_embeds = wan_compute_text_embeddings(
            prompts,
            text_encoders,
            tokenizers,
            max_sequence_length=512,
            device=accelerator.device,
        )
        prompt_ids = tokenizers[0](
            prompts,
            padding="max_length",
            max_length=512,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(accelerator.device)

        generator = None
        if getattr(cfg.sample, "same_latent", False):
            generator = create_generator(
                prompts, base_seed=epoch * 10000 + i, device=accelerator.device
            )

        with autocast():
            with torch.no_grad():
                videos, latents, log_probs, kls = wan_pipeline_with_logprob(
                    pipeline,
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=sample_neg_prompt_embeds,
                    num_inference_steps=cfg.sample.num_steps,
                    guidance_scale=cfg.sample.guidance_scale,
                    output_type="pt",
                    return_dict=False,
                    num_frames=cfg.frames,
                    height=cfg.height,
                    width=cfg.width,
                    kl_reward=cfg.sample.kl_reward,
                    generator=generator,
                    noise_level=cfg.sample.noise_level,
                    sde_type=cfg.sample.sde_type,
                    diffusion_clip=cfg.sample.diffusion_clip,
                    diffusion_clip_value=cfg.sample.diffusion_clip_value,
                )

            latents = torch.stack(latents, dim=1)
            log_probs = torch.stack(log_probs, dim=1)
            kls = torch.stack(kls, dim=1)
            kl = kls.detach()

            timesteps = pipeline.scheduler.timesteps.repeat(cfg.sample.batch_size, 1)

            rewards_future = executor.submit(
                reward_fn, videos, prompts, prompt_metadata, True
            )
            time.sleep(0)

            if accelerator.is_main_process and len(train_video_cache) < 5:
                train_video_cache.append(
                    {
                        "videos": videos.detach().cpu(),
                        "prompts": prompts.copy(),
                        "future": rewards_future,
                    }
                )

            samples.append(
                {
                    "prompt_ids": prompt_ids,
                    "prompt_embeds": prompt_embeds,
                    "negative_prompt_embeds": sample_neg_prompt_embeds,
                    "timesteps": timesteps,
                    "latents": latents[:, :-1],
                    "next_latents": latents[:, 1:],
                    "log_probs": log_probs,
                    "kl": kl,
                    "rewards": rewards_future,
                }
            )

    for sample in tqdm(
        samples,
        desc="Waiting for rewards",
        disable=not accelerator.is_local_main_process,
        position=0,
    ):
        rewards, _ = sample["rewards"].result()
        sample["rewards"] = {
            key: torch.as_tensor(value, device=accelerator.device).float()
            for key, value in rewards.items()
        }

    if accelerator.is_main_process and train_video_cache and epoch % 10 == 0:
        for cache in train_video_cache:
            rewards_val, _ = cache["future"].result()
            log_videos(
                "sample",
                cfg,
                accelerator,
                cache["videos"],
                cache["prompts"],
                rewards_val,
                global_step,
            )

    return samples
