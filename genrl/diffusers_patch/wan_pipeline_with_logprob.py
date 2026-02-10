import contextlib
import math
import random
from collections.abc import Callable
from typing import Any

import torch
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.pipelines.wan.pipeline_output import WanPipelineOutput
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils.torch_utils import randn_tensor


def sde_step_with_logprob(
    self: UniPCMultistepScheduler,
    model_output: torch.FloatTensor,
    timestep: float | torch.FloatTensor,
    sample: torch.FloatTensor,
    noise_level: float = 0.7,
    prev_sample: torch.FloatTensor | None = None,
    generator: torch.Generator | None = None,
    sde_type: str | None = "flow_sde",
    deterministic: bool = False,
    return_sqrt_dt_and_std_dev_t: bool = False,
    diffusion_clip: bool = False,
    diffusion_clip_value: float = 0.45,
):
    """
    Predict the sample from the previous timestep by reversing the SDE.

    Args:
        self: Scheduler.
        model_output: Predicted noise/velocity.
        timestep: Current timestep(s).
        sample: Current latents.
        noise_level: Noise level for SDE/CPS computation.
        prev_sample: Optional precomputed previous sample (mutually exclusive with generator).
        generator: Optional RNG for sampling prev_sample.
        sde_type: Type of SDE, either 'flow_sde' or 'flow_cps'.
        deterministic: If True, no noise added (deterministic update).
        return_sqrt_dt_and_std_dev_t: If True, also return std_dev_t and sqrt(-dt).
        diffusion_clip: If True, clip the std_dev_t to the diffusion_clip_value.
        diffusion_clip_value: Value to clip the std_dev_t to.

    Returns:
        If return_sqrt_dt_and_std_dev_t:
            (prev_sample, log_prob, prev_sample_mean, std_dev_t, sqrt_neg_dt, sigma, sigma_max)
        else:
            (prev_sample, log_prob, prev_sample_mean, std_dev_t * sqrt_neg_dt, sigma, sigma_max)
    """
    model_output = model_output.float()
    sample = sample.float()
    if prev_sample is not None:
        prev_sample = prev_sample.float()

    step_index = [self.index_for_timestep(t) for t in timestep]
    prev_step_index = [step + 1 for step in step_index]

    self.sigmas = self.sigmas.to(sample.device)
    sigma = self.sigmas[step_index].view(-1, 1, 1, 1, 1)
    sigma_prev = self.sigmas[prev_step_index].view(-1, 1, 1, 1, 1)
    sigma_max = self.sigmas[1].item()
    dt = sigma_prev - sigma

    if sde_type == "flow_sde":
        std_dev_t = (
            torch.sqrt(
                sigma
                / (
                    1
                    - torch.where(
                        sigma == 1,
                        torch.tensor(sigma_max, device=sigma.device, dtype=sigma.dtype),
                        sigma,
                    )
                )
            )
            * noise_level
        )

        if diffusion_clip:
            # https://arxiv.org/pdf/2510.22200: truncated noise schedule for flow_sde
            # std_dev_t and dt are tensors (batched); apply element-wise clipping.
            max_std_dev_t = diffusion_clip_value / torch.sqrt(-1 * dt)
            std_dev_t = torch.minimum(std_dev_t, max_std_dev_t)

        prev_sample_mean = (
            sample * (1 + std_dev_t**2 / (2 * sigma) * dt)
            + model_output * (1 + std_dev_t**2 * (1 - sigma) / (2 * sigma)) * dt
        )

        if prev_sample is None:
            variance_noise = randn_tensor(
                model_output.shape,
                generator=generator,
                device=model_output.device,
                dtype=model_output.dtype,
                # NOTE: I do not pass generator=generator here.
                # This is also reproducible, because I have set global seed in the trainer.
                # Some local seeding would not impact the global seed, and thus the reproducibility.
            )
            prev_sample = (
                prev_sample_mean + std_dev_t * torch.sqrt(-1 * dt) * variance_noise
            )

        if deterministic:
            prev_sample = sample + dt * model_output

        log_prob = (
            -((prev_sample.detach() - prev_sample_mean) ** 2)
            / (2 * ((std_dev_t * torch.sqrt(-1 * dt)) ** 2))
            - torch.log(std_dev_t * torch.sqrt(-1 * dt))
            - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
        )

    elif sde_type == "flow_cps":
        std_dev_t = sigma_prev * math.sin(noise_level * math.pi / 2)  # sigma_t in paper
        pred_original_sample = sample - sigma * model_output  # predicted x_0 in paper
        noise_estimate = sample + model_output * (1 - sigma)  # predicted x_1 in paper
        prev_sample_mean = pred_original_sample * (
            1 - sigma_prev
        ) + noise_estimate * torch.sqrt(sigma_prev**2 - std_dev_t**2)

        if prev_sample is None:
            variance_noise = randn_tensor(
                model_output.shape,
                generator=generator,
                device=model_output.device,
                dtype=model_output.dtype,
            )
            prev_sample = prev_sample_mean + std_dev_t * variance_noise

        if deterministic:
            prev_sample = (
                pred_original_sample * (1 - sigma_prev) + noise_estimate * sigma_prev
            )

        # remove all constants
        log_prob = -((prev_sample.detach() - prev_sample_mean) ** 2)

    else:
        msg = f"Unknown sde_type: {sde_type}. Must be 'flow_sde' or 'flow_cps'."
        raise ValueError(
            msg
        )

    # mean along all but batch dimension
    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

    if return_sqrt_dt_and_std_dev_t:
        return (
            prev_sample,
            log_prob,
            prev_sample_mean,
            std_dev_t,
            torch.sqrt(-1 * dt),
            sigma,
            sigma_max,
        )
    return (
        prev_sample,
        log_prob,
        prev_sample_mean,
        std_dev_t * torch.sqrt(-1 * dt),
        sigma,
        sigma_max,
    )


def wan_pipeline_with_logprob(
    self,
    prompt: str | list[str] | None = None,
    negative_prompt: str | list[str] | None = None,
    height: int = 480,
    width: int = 832,
    num_frames: int = 81,
    num_inference_steps: int = 50,
    guidance_scale: float = 5.0,
    num_videos_per_prompt: int | None = 1,
    generator: torch.Generator | list[torch.Generator] | None = None,
    latents: torch.Tensor | None = None,
    prompt_embeds: torch.Tensor | None = None,
    negative_prompt_embeds: torch.Tensor | None = None,
    output_type: str | None = "np",
    return_dict: bool = True,
    attention_kwargs: dict[str, Any] | None = None,
    callback_on_step_end: Callable[[int, int, dict], None] | PipelineCallback | MultiPipelineCallbacks | None = None,
    callback_on_step_end_tensor_inputs: list[str] | None = None,
    max_sequence_length: int = 512,
    deterministic: bool = False,
    kl_reward: float = 0.0,
    noise_level: float = 0.7,
    sde_type: str | None = "flow_sde",
    diffusion_clip: bool = False,
    diffusion_clip_value: float = 0.45,
    sde_window_size: int = 0,
    sde_window_range: tuple[int, int] | None = None,
):
    if callback_on_step_end_tensor_inputs is None:
        callback_on_step_end_tensor_inputs = ["latents"]
    if isinstance(callback_on_step_end, PipelineCallback | MultiPipelineCallbacks):
        callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

    self.check_inputs(
        prompt,
        negative_prompt,
        height,
        width,
        prompt_embeds,
        negative_prompt_embeds,
        callback_on_step_end_tensor_inputs,
    )

    if num_frames % self.vae_scale_factor_temporal != 1:
        num_frames = (
            num_frames
            // self.vae_scale_factor_temporal
            * self.vae_scale_factor_temporal
            + 1
        )
    num_frames = max(num_frames, 1)

    self._guidance_scale = guidance_scale
    self._attention_kwargs = attention_kwargs
    self._current_timestep = None
    self._interrupt = False

    device = self._execution_device

    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    prompt_embeds, negative_prompt_embeds = self.encode_prompt(
        prompt=prompt,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=self.do_classifier_free_guidance,
        num_videos_per_prompt=num_videos_per_prompt,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        max_sequence_length=max_sequence_length,
        device=device,
    )

    transformer_dtype = self.transformer.dtype
    prompt_embeds = prompt_embeds.to(transformer_dtype)
    if negative_prompt_embeds is not None:
        negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

    self.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = self.scheduler.timesteps

    num_channels_latents = self.transformer.config.in_channels
    latents = self.prepare_latents(
        batch_size * num_videos_per_prompt,
        num_channels_latents,
        height,
        width,
        num_frames,
        torch.float32,
        device,
        generator,
        latents,
    )

    # Determine SDE window for training
    # If sde_window_size > 0, randomly select a window within sde_window_range
    # If sde_window_size == 0, skip window logic and use original behavior
    use_window = sde_window_size > 0 and sde_window_range is not None
    if use_window:
        # Validate sde_window_range
        if sde_window_range[1] - sde_window_range[0] < sde_window_size:
            msg = (
                f"sde_window_range span ({sde_window_range[1] - sde_window_range[0]}) "
                f"must be >= sde_window_size ({sde_window_size})"
            )
            raise ValueError(
                msg
            )
        # Use generator if provided (for training reproducibility), otherwise fallback to random
        if generator is not None:
            # Extract generator from list if needed
            gen = generator[0] if isinstance(generator, list) and len(generator) > 0 else generator
            # Use torch.randint with generator for deterministic randomness
            max_start = sde_window_range[1] - sde_window_size
            start = torch.randint(
                sde_window_range[0], max_start + 1, (1,), generator=gen, device=device
            ).item()
        else:
            # Fallback to Python random (for eval, where generator may not be provided)
            # This is safe because eval uses deterministic=True and set_seed at the start
            start = random.randint(
                sde_window_range[0], sde_window_range[1] - sde_window_size
            )
        end = start + sde_window_size
        sde_window = (start, end)
        # In window mode, initialize all_latents as empty list (will be populated in the loop)
        # This matches flow_grpo's behavior
        all_latents: list[torch.Tensor] = []
    else:
        sde_window = None
        # In non-window mode, initialize all_latents with initial latent (matches git HEAD)
        all_latents: list[torch.Tensor] = [latents]

    all_log_probs: list[torch.Tensor] = []
    all_kl: list[torch.Tensor] = []
    all_timesteps: list[torch.Tensor] = []

    # 6. Denoising loop
    num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
    self._num_timesteps = len(timesteps)

    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            if self.interrupt:
                continue

            latents_ori = latents.clone()
            self._current_timestep = t
            latent_model_input = latents.to(transformer_dtype)
            timestep = t.expand(latents.shape[0])

            noise_pred = self.transformer(
                hidden_states=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                attention_kwargs=attention_kwargs,
                return_dict=False,
            )[0]
            noise_pred = noise_pred.to(prompt_embeds.dtype)

            if self.do_classifier_free_guidance:
                noise_uncond = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=negative_prompt_embeds,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0]
                noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)

            # Determine noise level based on SDE window
            if use_window:
                # Window mode: use noise_level only within the window, otherwise use 0
                if i < sde_window[0]:
                    cur_noise_level = 0.0
                elif i == sde_window[0]:
                    cur_noise_level = noise_level
                    # Record the initial latent at the start of the window
                    all_latents.append(latents)
                elif i > sde_window[0] and i < sde_window[1]:
                    cur_noise_level = noise_level
                else:
                    cur_noise_level = 0.0
            else:
                # Original mode: always use noise_level (sde_window_size == 0)
                cur_noise_level = noise_level

            (
                latents,
                log_prob,
                prev_latents_mean,
                std_dev_t,
                sigma,
                sigma_max,
            ) = sde_step_with_logprob(
                self.scheduler,
                noise_pred.float(),
                t.unsqueeze(0),
                latents.float(),
                noise_level=cur_noise_level,
                sde_type=sde_type,
                deterministic=deterministic,
                diffusion_clip=diffusion_clip,
                diffusion_clip_value=diffusion_clip_value,
            )
            prev_latents = latents.clone()

            # Record latents and log_probs
            if use_window:
                # Window mode: only record within the SDE window
                in_window = i >= sde_window[0] and i < sde_window[1]
                if in_window:
                    all_latents.append(latents)
                    all_log_probs.append(log_prob)
                    all_timesteps.append(t)
            else:
                # Original mode: record all timesteps (sde_window_size == 0)
                all_latents.append(latents)
                all_log_probs.append(log_prob)
                all_timesteps.append(t)

            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                negative_prompt_embeds = callback_outputs.pop(
                    "negative_prompt_embeds", negative_prompt_embeds
                )

            # Compute KL reward
            if use_window:
                # Window mode: only compute KL within the SDE window
                in_window = i >= sde_window[0] and i < sde_window[1]
                if in_window:
                    if kl_reward > 0 and not deterministic:
                        latent_model_input = (
                            torch.cat([latents_ori] * 2)
                            if self.do_classifier_free_guidance
                            else latents_ori
                        )
                        ref_model = getattr(self, "ref_transformer", None)
                        if ref_model is not None:
                            ref_ctx = contextlib.nullcontext()
                            target_model = ref_model
                        else:
                            target_model = self.transformer
                            ref_ctx = (
                                target_model.disable_adapter()
                                if hasattr(target_model, "disable_adapter")
                                else contextlib.nullcontext()
                            )

                        with ref_ctx:
                            noise_pred = target_model(
                                hidden_states=latent_model_input,
                                timestep=timestep,
                                encoder_hidden_states=prompt_embeds,
                                attention_kwargs=attention_kwargs,
                                return_dict=False,
                            )[0]
                        noise_pred = noise_pred.to(prompt_embeds.dtype)
                        # perform guidance
                        if self.do_classifier_free_guidance:
                            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                            noise_pred = noise_pred_uncond + self.guidance_scale * (
                                noise_pred_text - noise_pred_uncond
                            )

                        (
                            _,
                            ref_log_prob,
                            ref_prev_latents_mean,
                            ref_std_dev_t,
                            ref_sigma,
                            ref_sigma_max,
                        ) = sde_step_with_logprob(
                            self.scheduler,
                            noise_pred.float(),
                            t.unsqueeze(0),
                            latents_ori.float(),
                            noise_level=noise_level,
                            sde_type=sde_type,
                            prev_sample=prev_latents.float(),
                            deterministic=deterministic,
                            diffusion_clip=diffusion_clip,
                            diffusion_clip_value=diffusion_clip_value,
                        )
                        assert std_dev_t == ref_std_dev_t
                        kl = (prev_latents_mean - ref_prev_latents_mean) ** 2 / (
                            2 * std_dev_t**2
                        )
                        kl = kl.mean(dim=tuple(range(1, kl.ndim)))
                        all_kl.append(kl)
                    else:
                        # In window but no KL reward, append zero KL
                        all_kl.append(torch.zeros(len(latents), device=latents.device))
            # Original mode: compute KL for all timesteps (sde_window_size == 0)
            elif kl_reward > 0 and not deterministic:
                latent_model_input = (
                    torch.cat([latents_ori] * 2)
                    if self.do_classifier_free_guidance
                    else latents_ori
                )
                ref_model = getattr(self, "ref_transformer", None)
                if ref_model is not None:
                    ref_ctx = contextlib.nullcontext()
                    target_model = ref_model
                else:
                    target_model = self.transformer
                    ref_ctx = (
                        target_model.disable_adapter()
                        if hasattr(target_model, "disable_adapter")
                        else contextlib.nullcontext()
                    )

                with ref_ctx:
                    noise_pred = target_model(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=prompt_embeds,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                    )[0]
                noise_pred = noise_pred.to(prompt_embeds.dtype)
                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                (
                    _,
                    ref_log_prob,
                    ref_prev_latents_mean,
                    ref_std_dev_t,
                    ref_sigma,
                    ref_sigma_max,
                ) = sde_step_with_logprob(
                    self.scheduler,
                    noise_pred.float(),
                    t.unsqueeze(0),
                    latents_ori.float(),
                    noise_level=noise_level,
                    sde_type=sde_type,
                    prev_sample=prev_latents.float(),
                    deterministic=deterministic,
                    diffusion_clip=diffusion_clip,
                    diffusion_clip_value=diffusion_clip_value,
                )
                assert std_dev_t == ref_std_dev_t
                kl = (prev_latents_mean - ref_prev_latents_mean) ** 2 / (
                    2 * std_dev_t**2
                )
                kl = kl.mean(dim=tuple(range(1, kl.ndim)))
                all_kl.append(kl)
            else:
                # no kl reward, we do not need to compute, just put a pre-position value, kl will be 0
                all_kl.append(torch.zeros(len(latents), device=latents.device))

            # call the callback, if provided
            if i == len(timesteps) - 1 or (
                (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
            ):
                progress_bar.update()

    self._current_timestep = None

    if output_type != "latent":
        latents = latents.to(self.vae.dtype)
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(
            1, self.vae.config.z_dim, 1, 1, 1
        ).to(latents.device, latents.dtype)
        latents = latents / latents_std + latents_mean
        # Decode one sample at a time to reduce peak memory.
        decoded_videos = []
        for idx in range(latents.shape[0]):
            decoded = self.vae.decode(latents[idx : idx + 1], return_dict=False)[0]
            decoded_videos.append(decoded)
        video = torch.cat(decoded_videos, dim=0)
        video = self.video_processor.postprocess_video(video, output_type=output_type)
    else:
        video = latents

    self.maybe_free_model_hooks()

    if not return_dict:
        return (video, all_latents, all_log_probs, all_kl, all_timesteps)

    return (
        WanPipelineOutput(frames=video),
        all_latents,
        all_log_probs,
        all_kl,
        all_timesteps,
    )
