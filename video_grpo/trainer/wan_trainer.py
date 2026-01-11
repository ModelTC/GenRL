import os
import datetime
import time
import json
import copy
import numpy as np
import contextlib
from collections import defaultdict
from concurrent import futures
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import tqdm
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import WanPipeline
from peft import LoraConfig, get_peft_model, PeftModel
from loguru import logger

from video_grpo.config import Config
from video_grpo.ema import EMAModuleWrapper
from video_grpo.data import build_dataloaders
from video_grpo.stat_tracking import PerPromptStatTracker
from video_grpo.rewards import multi_score
from video_grpo.diffusers_patch.wan_pipeline_with_logprob import (
    wan_pipeline_with_logprob,
    sde_step_with_logprob,
)
from video_grpo.diffusers_patch.wan_prompt_embedding import encode_prompt
from video_grpo.utils import (  # type: ignore
    build_accelerator,
    resolve_resume_checkpoint,
    log_videos,
    save_ckpt,
    calculate_zero_std_ratio,
    create_generator,
    fast_init,
)

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)


def compute_text_embeddings(
    prompt: str | List[str],
    text_encoders: List[Any],
    tokenizers: List[Any],
    max_sequence_length: int,
    device: torch.device,
) -> torch.FloatTensor:
    """Encode prompts into embeddings on target device.

    Args:
        prompt: String or list of strings to encode.
        text_encoders: Sequence of text encoder modules.
        tokenizers: Sequence of tokenizers aligned with encoders.
        max_sequence_length: Max token length for encoding.
        device: Target device for embeddings.

    Returns:
        Tensor of encoded prompt embeddings on `device`.
    """
    with torch.no_grad():
        prompt_embeds = encode_prompt(
            text_encoders, tokenizers, prompt, max_sequence_length
        )
        prompt_embeds = prompt_embeds.to(device)
    return prompt_embeds


def compute_log_prob(
    transformer: torch.nn.Module,
    pipeline: WanPipeline,
    sample: Dict[str, torch.Tensor],
    j: int,
    embeds: torch.Tensor,
    negative_embeds: Optional[torch.Tensor],
    cfg: Config,
    attention_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        Tuple (prev_sample, log_prob, prev_sample_mean, std_dev_t, dt).
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
        noise_pred = noise_pred_uncond + cfg.sample.guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )
    else:
        noise_pred = transformer(
            hidden_states=sample["latents"][:, j],
            timestep=sample["timesteps"][:, j],
            encoder_hidden_states=embeds,
            return_dict=False,
        )[0]

    prev_sample, log_prob, prev_sample_mean, std_dev_t, dt = sde_step_with_logprob(
        pipeline.scheduler,
        noise_pred.float(),
        sample["timesteps"][:, j],
        sample["latents"][:, j].float(),
        prev_sample=sample["next_latents"][:, j].float(),
        return_dt_and_std_dev_t=True,
    )
    return prev_sample, log_prob, prev_sample_mean, std_dev_t, dt


def eval_once(
    cfg: Config,
    accelerator: Accelerator,
    pipeline: WanPipeline,
    test_dataloader: torch.utils.data.DataLoader,
    text_encoders: List[Any],
    tokenizers: List[Any],
    sample_neg_prompt_embeds: torch.Tensor,
    eval_reward_fn: Callable,
    autocast: Any,
    global_step: int,
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

    Returns:
        None. Logs metrics/videos to accelerator.
    """
    set_seed(cfg.seed, device_specific=True)
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
        test_embeds = compute_text_embeddings(
            test_prompts,
            text_encoders,
            tokenizers,
            max_sequence_length=512,
            device=accelerator.device,
        )
        with autocast():
            with torch.no_grad():
                videos_eval, _, _, _ = wan_pipeline_with_logprob(
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
                    determistic=True,
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
    accelerator.log(
        {
            **{
                f"eval_reward_{k}": np.mean(v[v != -10]) for k, v in all_rewards.items()
            },
        },
        step=global_step,
    )
    if accelerator.is_main_process and last_batch is not None:
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


def sample_epoch(
    cfg: Config,
    accelerator: Accelerator,
    pipeline: WanPipeline,
    train_sampler: Any,
    train_iter,
    reward_fn: Callable,
    sample_neg_prompt_embeds: torch.Tensor,
    text_encoders: List[Any],
    tokenizers: List[Any],
    executor: futures.ThreadPoolExecutor,
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

        prompt_embeds = compute_text_embeddings(
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


def train(cfg: Config):
    """Main training loop for VideoGRPO.

    Args:
        cfg: Parsed Config object.

    Returns:
        None. Executes training, logging, and checkpointing side effects.
    """
    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    cfg.run_name = cfg.run_name + "_" + unique_id if cfg.run_name else unique_id
    resume_path = resolve_resume_checkpoint(getattr(cfg.paths, "resume_from", None))

    num_train_timesteps = int(cfg.sample.num_steps * cfg.train.timestep_fraction)
    # derive base grad accumulation if not provided: align with original sampler settings
    base_gas = cfg.train.gradient_accumulation_steps
    if base_gas is None or base_gas <= 0:
        total_chunks = cfg.sample.num_batches_per_epoch
        base_gas = total_chunks // 2 if total_chunks > 1 else 1
    cfg.train.gradient_accumulation_steps = base_gas
    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(cfg.paths.logdir, cfg.run_name),
        automatic_checkpoint_naming=True,
        total_limit=cfg.num_checkpoint_limit,
    )
    train_timesteps = [step_index for step_index in range(num_train_timesteps)]
    gradient_accumulation_steps = base_gas * num_train_timesteps

    accelerator = build_accelerator(
        cfg, gradient_accumulation_steps, accelerator_config
    )

    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name=cfg.project_name,
            config=cfg.__dict__,
            init_kwargs={"wandb": {"name": cfg.run_name}},
        )
    logger.info(f"\n{cfg}")

    set_seed(cfg.seed, device_specific=True)

    with fast_init(accelerator.device, init_weights=False):
        pipeline = WanPipeline.from_pretrained(cfg.paths.pretrained_model)
    # mutually exclusive: use_lora -> LoRA path; otherwise full finetune
    full_finetune = not cfg.use_lora
    ref_transformer = None
    if full_finetune and cfg.train.beta > 0:
        # keep a frozen ref model; deepcopy to mirror original behavior
        ref_transformer = copy.deepcopy(pipeline.transformer)
        ref_transformer.requires_grad_(False)

    if full_finetune:
        cfg.use_lora = False
        pipeline.vae.requires_grad_(False)
        pipeline.text_encoder.requires_grad_(False)
        pipeline.transformer.requires_grad_(True)
    else:
        pipeline.vae.requires_grad_(False)
        pipeline.text_encoder.requires_grad_(False)
        pipeline.transformer.requires_grad_(False)

    text_encoders = [pipeline.text_encoder]
    tokenizers = [pipeline.tokenizer]

    pipeline.safety_checker = None
    pipeline.set_progress_bar_config(
        position=1,
        disable=not accelerator.is_local_main_process,
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )

    inference_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16

    pipeline.vae.to(accelerator.device, dtype=torch.float32)
    pipeline.text_encoder.to(accelerator.device, dtype=inference_dtype)

    if cfg.use_lora:
        pipeline.transformer.to(accelerator.device)
        transformer_lora_config = LoraConfig(
            r=cfg.train.lora_r,
            lora_alpha=cfg.train.lora_alpha,
            init_lora_weights="gaussian",
            target_modules=cfg.train.lora_target_modules,
        )
        if cfg.train.lora_path:
            pipeline.transformer = PeftModel.from_pretrained(
                pipeline.transformer, cfg.train.lora_path
            )
            pipeline.transformer.set_adapter("default")
        else:
            pipeline.transformer = get_peft_model(
                pipeline.transformer, transformer_lora_config
            )

    transformer = pipeline.transformer
    transformer.enable_gradient_checkpointing()
    trainable_modules = [transformer]
    if full_finetune:
        trainable_modules.extend([pipeline.vae, pipeline.text_encoder])
    transformer_params = []
    for module in trainable_modules:
        transformer_params.extend(
            list(filter(lambda p: p.requires_grad, module.parameters()))
        )
    ema = EMAModuleWrapper(
        transformer_params,
        decay=cfg.train.ema_decay,
        update_step_interval=cfg.train.ema_update_interval,
        device=accelerator.device,
    )

    if cfg.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if cfg.train.use_8bit_adam:
        import bitsandbytes as bnb

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        transformer_params,
        lr=cfg.train.learning_rate,
        betas=(cfg.train.adam_beta1, cfg.train.adam_beta2),
        weight_decay=cfg.train.adam_weight_decay,
        eps=cfg.train.adam_epsilon,
    )

    reward_fn = multi_score(accelerator.device, cfg.reward_fn, cfg.reward_module)
    eval_reward_cfg = (
        cfg.eval_reward_fn if cfg.eval_reward_fn is not None else cfg.reward_fn
    )
    eval_reward_module = (
        cfg.eval_reward_module
        if cfg.eval_reward_module is not None
        else cfg.reward_module
    )
    eval_reward_fn = multi_score(
        accelerator.device, eval_reward_cfg, eval_reward_module
    )

    train_dataloader, test_dataloader, train_sampler = build_dataloaders(
        cfg, accelerator
    )

    neg_prompt_embed = compute_text_embeddings(
        [""],
        text_encoders,
        tokenizers,
        max_sequence_length=512,
        device=accelerator.device,
    )
    sample_neg_prompt_embeds = neg_prompt_embed.repeat(cfg.sample.batch_size, 1, 1)
    train_neg_prompt_embeds = neg_prompt_embed.repeat(cfg.train.batch_size, 1, 1)

    if cfg.sample.num_video_per_prompt == 1:
        cfg.per_prompt_stat_tracking = False
    if cfg.per_prompt_stat_tracking:
        stat_tracker = PerPromptStatTracker(cfg.sample.global_std)

    autocast = contextlib.nullcontext if cfg.use_lora else accelerator.autocast

    # async reward executor
    executor = futures.ThreadPoolExecutor(max_workers=8)

    transformer, optimizer, test_dataloader = accelerator.prepare(
        transformer, optimizer, test_dataloader
    )
    if ref_transformer is not None:
        ref_transformer = accelerator.prepare_model(
            ref_transformer, evaluation_mode=True
        )
        ref_transformer.eval()
        pipeline.ref_transformer = ref_transformer

    train_iter = iter(train_dataloader)
    first_epoch = 0
    global_step = 0
    resume_epoch_tag = None
    if resume_path:
        logger.info(f"Resuming from {resume_path}")
        accelerator.load_state(resume_path)
        meta_path = os.path.join(resume_path, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                metadata = json.load(f)
            global_step = metadata.get("global_step", 0)
            first_epoch = metadata.get("epoch", 0)
            resume_epoch_tag = metadata.get("current_epoch_tag", None)
        if cfg.train.ema:
            ema_state_path = os.path.join(resume_path, "ema_state.pt")
            if os.path.exists(ema_state_path):
                ema_state = torch.load(ema_state_path, map_location=accelerator.device)
                ema.load_state_dict(ema_state)
                ema.to(accelerator.device)
    if resume_epoch_tag is not None:
        train_sampler.set_epoch(resume_epoch_tag)

    samples_per_epoch = (
        cfg.sample.batch_size
        * accelerator.num_processes
        * cfg.sample.num_batches_per_epoch
    )
    total_train_batch_size = (
        cfg.train.batch_size
        * accelerator.num_processes
        * cfg.train.gradient_accumulation_steps
    )

    logger.info(
        "\n".join(
            [
                "***** Running training *****",
                f"  Num Epochs = {cfg.num_epochs}",
                f"  Sample batch size per device = {cfg.sample.batch_size}",
                f"  Train batch size per device = {cfg.train.batch_size}",
                f"  Gradient Accumulation steps = {cfg.train.gradient_accumulation_steps}",
                f"  Total number of samples per epoch = {samples_per_epoch}",
                f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}",
                f"  Number of gradient updates per inner epoch = {samples_per_epoch // total_train_batch_size}",
                f"  Number of inner epochs = {cfg.train.num_inner_epochs}",
            ]
        )
    )

    for epoch in range(first_epoch, cfg.num_epochs):
        pipeline.transformer.eval()

        if epoch % cfg.eval_freq == 0 and epoch > 0:
            eval_once(
                cfg,
                accelerator,
                pipeline,
                test_dataloader,
                text_encoders,
                tokenizers,
                sample_neg_prompt_embeds,
                eval_reward_fn,
                autocast,
                global_step,
            )
        if epoch % cfg.save_freq == 0 and epoch > 0 and accelerator.is_main_process:
            current_epoch_tag = epoch * cfg.sample.num_batches_per_epoch
            save_ckpt(
                cfg,
                transformer,
                global_step,
                epoch,
                accelerator,
                ema,
                transformer_params,
                current_epoch_tag,
            )
        accelerator.wait_for_everyone()

        samples = sample_epoch(
            cfg,
            accelerator,
            pipeline,
            train_sampler,
            train_iter,
            reward_fn,
            sample_neg_prompt_embeds,
            text_encoders,
            tokenizers,
            executor,
            autocast,
            epoch,
            global_step,
        )

        samples["rewards"]["ori_avg"] = samples["rewards"]["avg"]
        samples["rewards"]["avg"] = (
            samples["rewards"]["avg"].unsqueeze(-1)
            - cfg.sample.kl_reward * samples["kl"]
        )
        gathered_rewards = {
            key: accelerator.gather(value) for key, value in samples["rewards"].items()
        }
        gathered_rewards = {
            key: value.cpu().numpy() for key, value in gathered_rewards.items()
        }

        # log rewards and KL (mirror original script behavior)
        reward_logs = {
            f"reward_{key}": value.mean()
            for key, value in gathered_rewards.items()
            if "_strict_accuracy" not in key and "_accuracy" not in key
        }
        kl_mean = samples["kl"].mean().detach().cpu().item()
        kl_abs = samples["kl"].abs().mean().detach().cpu().item()
        accelerator.log(
            {
                "epoch": epoch,
                **reward_logs,
                "kl": kl_mean,
                "kl_abs": kl_abs,
            },
            step=global_step,
        )

        if cfg.per_prompt_stat_tracking:
            prompt_ids = accelerator.gather(samples["prompt_ids"]).cpu().numpy()
            prompts = pipeline.tokenizer.batch_decode(
                prompt_ids, skip_special_tokens=True
            )
            advantages = stat_tracker.update(prompts, gathered_rewards["avg"])
            if accelerator.is_local_main_process:
                logger.info(
                    f"len(prompts) {len(prompts)} | len unique {len(set(prompts))}"
                )
            group_size, trained_prompt_num = stat_tracker.get_stats()
            zero_std_ratio = calculate_zero_std_ratio(prompts, gathered_rewards)
            accelerator.log(
                {
                    "group_size": group_size,
                    "trained_prompt_num": trained_prompt_num,
                    "zero_std_ratio": zero_std_ratio,
                },
                step=global_step,
            )
            stat_tracker.clear()
        else:
            advantages = (gathered_rewards["avg"] - gathered_rewards["avg"].mean()) / (
                gathered_rewards["avg"].std() + 1e-4
            )

        advantages = torch.as_tensor(advantages)
        samples["advantages"] = advantages.reshape(
            accelerator.num_processes, -1, advantages.shape[-1]
        )[accelerator.process_index].to(accelerator.device)
        del samples["rewards"]
        del samples["prompt_ids"]

        mask = samples["advantages"].abs().sum(dim=1) != 0
        num_batches = cfg.sample.num_batches_per_epoch
        true_count = mask.sum()
        if true_count == 0:
            samples["advantages"] = samples["advantages"] + 1e-6
            mask = samples["advantages"].abs().sum(dim=1) != 0
        if true_count % num_batches != 0:
            false_indices = torch.where(~mask)[0]
            num_to_change = num_batches - (true_count % num_batches)
            if len(false_indices) >= num_to_change:
                random_indices = torch.randperm(len(false_indices))[:num_to_change]
                mask[false_indices[random_indices]] = True
        accelerator.log(
            {
                "actual_batch_size": mask.sum().item() // num_batches,
            },
            step=global_step,
        )
        samples = {k: v[mask] for k, v in samples.items()}

        total_batch_size, num_timesteps = samples["timesteps"].shape

        for inner_epoch in range(cfg.train.num_inner_epochs):
            perm = torch.randperm(total_batch_size, device=accelerator.device)
            samples = {k: v[perm] for k, v in samples.items()}
            perms = torch.stack(
                [
                    torch.arange(num_timesteps, device=accelerator.device)
                    for _ in range(total_batch_size)
                ]
            )
            for key in ["timesteps", "latents", "next_latents", "log_probs"]:
                samples[key] = samples[key][
                    torch.arange(total_batch_size, device=accelerator.device)[:, None],
                    perms,
                ]

            micoe_batch = total_batch_size // cfg.sample.num_batches_per_epoch
            samples_batched = {
                k: v.reshape(-1, micoe_batch, *v.shape[1:]) for k, v in samples.items()
            }
            samples_batched = [
                dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())
            ]

            pipeline.transformer.train()
            info = defaultdict(list)
            for i, sample in tqdm(
                list(enumerate(samples_batched)),
                desc=f"Epoch {epoch}.{inner_epoch}: training",
                position=0,
                disable=not accelerator.is_local_main_process,
            ):
                if cfg.train.cfg:
                    embeds = sample["prompt_embeds"]
                    negative_embeds = train_neg_prompt_embeds[
                        : len(sample["prompt_embeds"])
                    ]
                else:
                    embeds = sample["prompt_embeds"]
                    negative_embeds = None

                for j in tqdm(
                    train_timesteps,
                    desc="Timestep",
                    position=1,
                    leave=False,
                    disable=not accelerator.is_local_main_process,
                ):
                    with accelerator.accumulate(transformer):
                        with autocast():
                            prev_sample, log_prob, prev_sample_mean, std_dev_t, dt = (
                                compute_log_prob(
                                    transformer,
                                    pipeline,
                                    sample,
                                    j,
                                    embeds,
                                    negative_embeds,
                                    cfg,
                                )
                            )
                            if cfg.train.beta > 0:
                                if full_finetune:
                                    ref_model = ref_transformer
                                    if ref_model is None:
                                        raise ValueError(
                                            "full_finetune with beta>0 requires a ref_transformer."
                                        )
                                    ref_model.eval()
                                    with torch.no_grad():
                                        (
                                            prev_sample_ref,
                                            log_prob_ref,
                                            prev_sample_mean_ref,
                                            std_dev_t_ref,
                                            dt_ref,
                                        ) = compute_log_prob(
                                            ref_model,
                                            pipeline,
                                            sample,
                                            j,
                                            embeds,
                                            negative_embeds,
                                            cfg,
                                        )
                                else:
                                    with torch.no_grad():
                                        with transformer.module.disable_adapter():
                                            (
                                                prev_sample_ref,
                                                log_prob_ref,
                                                prev_sample_mean_ref,
                                                std_dev_t_ref,
                                                dt_ref,
                                            ) = compute_log_prob(
                                                transformer,
                                                pipeline,
                                                sample,
                                                j,
                                                embeds,
                                                negative_embeds,
                                                cfg,
                                            )

                        advantages = torch.clamp(
                            sample["advantages"][:, j],
                            -cfg.train.adv_clip_max,
                            cfg.train.adv_clip_max,
                        )
                        ratio = torch.exp(log_prob - sample["log_probs"][:, j])
                        unclipped_loss = -advantages * ratio
                        clipped_loss = -advantages * torch.clamp(
                            ratio,
                            1.0 - cfg.train.clip_range,
                            1.0 + cfg.train.clip_range,
                        )
                        policy_loss = torch.mean(
                            torch.maximum(unclipped_loss, clipped_loss)
                        )

                        if cfg.train.beta > 0:
                            kl_loss = (
                                (prev_sample_mean - prev_sample_mean_ref) ** 2
                            ).mean(dim=(1, 2, 3), keepdim=True) / (
                                2 * (std_dev_t * dt_ref) ** 2
                            )
                            kl_loss = torch.mean(kl_loss)
                            loss = policy_loss + cfg.train.beta * kl_loss
                        else:
                            loss = policy_loss

                        info["approx_kl"].append(
                            0.5
                            * torch.mean((log_prob - sample["log_probs"][:, j]) ** 2)
                        )
                        info["clip_frac"].append(
                            torch.mean(
                                (torch.abs(ratio - 1.0) > cfg.train.clip_range).float()
                            )
                        )
                        info["clip_frac_gt_one"].append(
                            torch.mean(
                                (ratio - 1.0 > cfg.train.clip_range).float()
                            )
                        )
                        info["clip_frac_lt_one"].append(
                            torch.mean(
                                (1.0 - ratio > cfg.train.clip_range).float()
                            )
                        )
                        info["policy_loss"].append(policy_loss)
                        if cfg.train.beta > 0:
                            info["kl_loss"].append(kl_loss)
                        info["loss"].append(loss)

                        accelerator.backward(loss)
                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(
                                transformer.parameters(), cfg.train.max_grad_norm
                            )
                        optimizer.step()
                        optimizer.zero_grad()
                    if accelerator.sync_gradients:
                        info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                        info = accelerator.reduce(info, reduction="mean")
                        info.update({"epoch": epoch, "inner_epoch": inner_epoch})
                        accelerator.log(info, step=global_step)
                        global_step += 1
                        info = defaultdict(list)
                if cfg.train.ema:
                    ema.step(transformer_params, global_step)


def run(cfg: Config):
    """Trainer entry used by factory.

    Args:
        cfg: Parsed Config object.

    Returns:
        None. Delegates to `train`.
    """
    train(cfg)
