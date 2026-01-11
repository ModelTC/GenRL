from typing import Any, Callable, Dict, List, Optional, Union
import torch
import contextlib
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils.torch_utils import randn_tensor
import math
import numpy as np
from diffusers.pipelines.wan import WanPipelineOutput


def sde_step_with_logprob(
    self: UniPCMultistepScheduler,
    model_output: torch.FloatTensor,
    timestep: Union[float, torch.FloatTensor],
    sample: torch.FloatTensor,
    prev_sample: Optional[torch.FloatTensor] = None,
    generator: Optional[torch.Generator] = None,
    determistic: bool = False,
    return_dt_and_std_dev_t: bool = False,
):
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
    sigma_min = self.sigmas[-1].item()
    dt = sigma_prev - sigma

    std_dev_t = sigma_min + (sigma_max - sigma_min) * sigma
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
        )
        prev_sample = (
            prev_sample_mean + std_dev_t * torch.sqrt(-1 * dt) * variance_noise
        )

    if determistic:
        prev_sample = sample + dt * model_output

    log_prob = (
        -((prev_sample.detach() - prev_sample_mean) ** 2)
        / (2 * ((std_dev_t * torch.sqrt(-1 * dt)) ** 2))
        - torch.log(std_dev_t * torch.sqrt(-1 * dt))
        - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
    )
    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

    out = (prev_sample, log_prob)
    if return_dt_and_std_dev_t:
        out = (prev_sample, log_prob, prev_sample_mean, std_dev_t, dt)
    return out


def wan_pipeline_with_logprob(
    self,
    prompt: Union[str, List[str]] = None,
    height: int = 480,
    width: int = 832,
    num_frames: int = 81,
    num_inference_steps: int = 50,
    guidance_scale: float = 5.0,
    num_videos_per_prompt: int = 1,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
    callback_on_step_end_tensor_inputs: List[str] = [
        "latents",
        "prompt_embeds",
        "negative_prompt_embeds",
    ],
    max_sequence_length: int = 226,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    generator: Optional[torch.Generator] = None,
    latents: Optional[torch.FloatTensor] = None,
    kl_reward: float = 0.0,
    determistic: bool = False,
    autocast_dtype: Optional[torch.dtype] = torch.bfloat16,
):
    if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
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
        prompt_embeds.dtype,
        device,
        generator,
        latents,
        num_frames=num_frames,
    )

    all_latents = []
    all_log_probs = []
    all_kl = []

    def get_noise_pred(latents, prompt_embeds, negative_prompt_embeds, timestep):
        noise_pred = self.transformer(
            hidden_states=latents,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            attention_kwargs=attention_kwargs,
            return_dict=False,
        )[0]
        noise_pred = noise_pred.to(prompt_embeds.dtype)
        if self.do_classifier_free_guidance:
            noise_uncond = self.transformer(
                hidden_states=latents,
                timestep=timestep,
                encoder_hidden_states=negative_prompt_embeds,
                attention_kwargs=attention_kwargs,
                return_dict=False,
            )[0]
            noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)
        return noise_pred

    with torch.autocast(device.type, dtype=autocast_dtype):
        with self.progress_bar(total=len(timesteps)) as progress_bar:
            for i, t in enumerate(timesteps):
                latents_ori = latents.clone()
                self._current_timestep = t
                latent_model_input = latents.to(transformer_dtype)
                timestep = t.expand(latents.shape[0])

                noise_pred = get_noise_pred(
                    latent_model_input, prompt_embeds, negative_prompt_embeds, timestep
                )

                latents, log_prob, prev_latents_mean, std_dev_t = sde_step_with_logprob(
                    self.scheduler,
                    noise_pred.float(),
                    t.unsqueeze(0),
                    latents.float(),
                    determistic=determistic,
                )
                prev_latents = latents.clone()

                all_latents.append(latents)
                all_log_probs.append(log_prob)

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

                if kl_reward > 0 and not determistic:
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
                        noise_pred_ref = target_model(
                            hidden_states=latent_model_input,
                            timestep=timestep,
                            encoder_hidden_states=prompt_embeds,
                            attention_kwargs=attention_kwargs,
                            return_dict=False,
                        )[0]
                    noise_pred_ref = noise_pred_ref.to(prompt_embeds.dtype)
                    if self.do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred_ref.chunk(2)
                        noise_pred_ref = noise_pred_uncond + self.guidance_scale * (
                            noise_pred_text - noise_pred_uncond
                        )

                    _, ref_log_prob, ref_prev_latents_mean, ref_std_dev_t = (
                        sde_step_with_logprob(
                            self.scheduler,
                            noise_pred_ref.float(),
                            t.unsqueeze(0),
                            latents_ori.float(),
                            prev_sample=prev_latents.float(),
                            determistic=determistic,
                        )
                    )
                    kl = (prev_latents_mean - ref_prev_latents_mean) ** 2 / (
                        2 * ref_std_dev_t**2
                    )
                    kl = kl.mean(dim=tuple(range(1, kl.ndim)))
                    all_kl.append(kl)
                else:
                    all_kl.append(torch.zeros(len(latents), device=latents.device))

                if i == len(timesteps) - 1 or (
                    (i + 1) > len(timesteps) // self.scheduler.order
                    and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()

    self._current_timestep = None

    if not output_type == "latent":
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
        video = self.vae.decode(latents, return_dict=False)[0]
        video = self.video_processor.postprocess_video(video, output_type=output_type)
    else:
        video = latents

    self.maybe_free_model_hooks()

    if not return_dict:
        return (video, all_latents, all_log_probs, all_kl)

    return WanPipelineOutput(frames=video), all_latents, all_log_probs, all_kl
