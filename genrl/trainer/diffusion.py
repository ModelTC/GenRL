"""Diffusion step computation utilities for trainers."""

from typing import Any

import torch

from genrl.config import Config
from genrl.diffusers_patch.wan_pipeline_with_logprob import sde_step_with_logprob


def wan_compute_log_prob(
    transformer: torch.nn.Module,
    pipeline: Any,  # Pipeline with scheduler
    sample: dict[str, torch.Tensor],
    j: int,
    embeds: torch.Tensor,
    negative_embeds: torch.Tensor | None,
    cfg: Config,
    attention_kwargs: dict[str, Any] | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    float,
]:
    """Run one diffusion step and return prev_sample stats and log prob.

    Args:
        transformer: Policy transformer (possibly wrapped).
        pipeline: Diffusion pipeline with scheduler.
        sample: Dict containing latents, next_latents, timesteps, log_probs.
        j: Timestep index.
        embeds: Conditional embeddings.
        negative_embeds: Unconditional embeddings (or None).
        cfg: Training config.
        attention_kwargs: Optional attention kwargs override.

    Returns:
        Tuple (prev_sample, log_prob, prev_sample_mean, std_dev_t, dt_sqrt, sigma, sigma_max).
    """
    attention_kwargs = attention_kwargs or getattr(cfg, "attention_kwargs", None)
    if cfg.train.cfg:
        noise_pred_text = transformer(
            hidden_states=sample["latents"][:, j],
            timestep=sample["timesteps"][:, j],
            encoder_hidden_states=embeds,
            attention_kwargs=attention_kwargs,
            return_dict=False,
        )[0]
        noise_pred_uncond = transformer(
            hidden_states=sample["latents"][:, j],
            timestep=sample["timesteps"][:, j],
            encoder_hidden_states=negative_embeds,
            attention_kwargs=attention_kwargs,
            return_dict=False,
        )[0]
        noise_pred = noise_pred_uncond + cfg.sample.guidance_scale * (noise_pred_text - noise_pred_uncond)
    else:
        noise_pred = transformer(
            hidden_states=sample["latents"][:, j],
            timestep=sample["timesteps"][:, j],
            encoder_hidden_states=embeds,
            return_dict=False,
        )[0]

    (
        prev_sample,
        log_prob,
        prev_sample_mean,
        std_dev_t,
        dt_sqrt,
        sigma,
        sigma_max,
    ) = sde_step_with_logprob(
        pipeline.scheduler,
        noise_pred.float(),
        sample["timesteps"][:, j],
        sample["latents"][:, j].float(),
        noise_level=cfg.sample.noise_level,
        prev_sample=sample["next_latents"][:, j].float(),
        sde_type=cfg.sample.sde_type,
        diffusion_clip=cfg.sample.diffusion_clip,
        diffusion_clip_value=cfg.sample.diffusion_clip_value,
        return_sqrt_dt_and_std_dev_t=True,
    )
    return prev_sample, log_prob, prev_sample_mean, std_dev_t, dt_sqrt, sigma, sigma_max
