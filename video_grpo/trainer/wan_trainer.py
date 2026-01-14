import os
import json
import copy
import contextlib
from collections import defaultdict
from concurrent import futures
from functools import partial
from typing import Any, Callable, Dict, List

import torch
import tqdm
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import WanPipeline
from peft import LoraConfig, get_peft_model, PeftModel
from loguru import logger

from video_grpo.config import Config
from video_grpo.constants import ADVANTAGE_EPSILON, SEED_EPOCH_STRIDE
from video_grpo.ema import EMAModuleWrapper
from video_grpo.data import build_dataloaders
from video_grpo.stat_tracking import PerPromptStatTracker
from video_grpo.rewards import multi_score
from video_grpo.advantages import compute_advantages
from video_grpo.trainer.sampling import wan_sample_epoch
from video_grpo.trainer.evaluation import wan_eval_once
from video_grpo.trainer.diffusion import wan_compute_log_prob
from video_grpo.trainer.embeddings import wan_compute_text_embeddings
from video_grpo.trainer.base_trainer import BaseTrainer
from video_grpo.utils import (  # type: ignore
    unwrap_model,
    fast_init,
)

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)


class WanTrainer(BaseTrainer):
    """WAN trainer for VideoGRPO training."""

    def __init__(self, cfg: Config):
        """Initialize WAN trainer.

        Args:
            cfg: Parsed Config object.
        """
        super().__init__(cfg)
        # Training state variables
        self.accelerator = None
        self.pipeline = None
        self.transformer = None
        self.ref_transformer = None
        self.optimizer = None
        self.ema = None
        self.train_dataloader = None
        self.test_dataloader = None
        self.train_sampler = None
        self.train_iter = None
        self.text_encoders = None
        self.tokenizers = None
        self.reward_fn = None
        self.eval_reward_fn = None
        self.stat_tracker = None
        self.reward_stat_trackers = None
        self.kl_stat_tracker = None
        self.sample_neg_prompt_embeds = None
        self.train_neg_prompt_embeds = None
        self.executor = None
        self.autocast = None
        self.train_timesteps = None
        self.full_finetune = None
        self.transformer_params = None

    def train(self):
        """Main training loop."""
        cfg = self.cfg

        # Setup training components
        # If sde_window_size > 0, use window size as num_train_timesteps
        # Otherwise, use timestep_fraction of total steps
        if cfg.sample.sde_window_size and cfg.sample.sde_window_size > 0:
            num_train_timesteps = cfg.sample.sde_window_size
        else:
            num_train_timesteps = int(
                cfg.sample.num_steps * cfg.train.timestep_fraction
            )
        (
            base_gas,
            gradient_accumulation_steps,
        ) = self.calculate_gradient_accumulation_steps(num_train_timesteps)
        train_timesteps = self.get_train_timesteps(num_train_timesteps)
        self.train_timesteps = train_timesteps

        accelerator = self.setup_accelerator(gradient_accumulation_steps)
        self.accelerator = accelerator

        # Setup pipeline and model
        self._setup_pipeline_and_model(cfg, accelerator)

        # Setup reward functions
        self._setup_reward_functions(cfg, accelerator)

        # Setup dataloaders
        self._setup_dataloaders(cfg, accelerator)

        # Setup stat trackers
        self._setup_stat_trackers(cfg)

        # Setup other components
        self._setup_training_utilities(cfg, accelerator)

        # Prepare models with accelerator
        self._prepare_models_with_accelerator(cfg, accelerator)

        # Resume from checkpoint if needed
        first_epoch, global_step, resume_epoch_tag = self.resume_from_checkpoint(
            accelerator
        )

        if resume_epoch_tag is not None:
            self.train_sampler.set_epoch(resume_epoch_tag)

        # Run training loop
        self._run_training_loop(cfg, accelerator, first_epoch, global_step)

    def _setup_pipeline_and_model(self, cfg: Config, accelerator: Accelerator):
        """Setup pipeline, model, and related components.

        Args:
            cfg: Training config.
            accelerator: Accelerator instance.
        """
        # Setup pipeline
        pipeline = self.setup_pipeline(WanPipeline, cfg.paths.pretrained_model)
        self.pipeline = pipeline

        # Determine finetune mode
        full_finetune = not cfg.use_lora
        self.full_finetune = full_finetune

        # Setup reference transformer for KL loss
        ref_transformer = None
        if full_finetune and cfg.train.beta > 0:
            ref_transformer = copy.deepcopy(pipeline.transformer)
            ref_transformer.requires_grad_(False)

        # Configure requires_grad
        if full_finetune:
            pipeline.vae.requires_grad_(False)
            pipeline.text_encoder.requires_grad_(False)
            pipeline.transformer.requires_grad_(True)
        else:
            pipeline.vae.requires_grad_(False)
            pipeline.text_encoder.requires_grad_(False)
            pipeline.transformer.requires_grad_(False)

        # Setup text encoders and tokenizers
        self.text_encoders = [pipeline.text_encoder]
        self.tokenizers = [pipeline.tokenizer]

        # Configure pipeline
        pipeline.safety_checker = None
        pipeline.set_progress_bar_config(
            position=1,
            disable=not accelerator.is_local_main_process,
            leave=False,
            desc="Timestep",
            dynamic_ncols=True,
        )

        # Setup mixed precision and move to device
        inference_dtype = self.setup_mixed_precision_dtype()
        pipeline.vae.to(accelerator.device, dtype=torch.float32)
        pipeline.text_encoder.to(accelerator.device, dtype=inference_dtype)

        # Setup LoRA if needed
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

        # Get transformer and collect trainable parameters
        transformer = pipeline.transformer
        self.transformer = transformer
        trainable_modules = [transformer]
        if full_finetune:
            trainable_modules.extend([pipeline.vae, pipeline.text_encoder])

        transformer_params = []
        for module in trainable_modules:
            transformer_params.extend(
                list(filter(lambda p: p.requires_grad, module.parameters()))
            )

        # Store ref_transformer for later use
        if ref_transformer is not None:
            self.ref_transformer = ref_transformer

        # Setup optimizer
        if cfg.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True

        optimizer = self.setup_optimizer(
            transformer_params, use_8bit=cfg.train.use_8bit_adam
        )
        self.optimizer = optimizer
        self.transformer_params = transformer_params

    def _setup_reward_functions(self, cfg: Config, accelerator: Accelerator):
        """Setup reward functions for training and evaluation.

        Args:
            cfg: Training config.
            accelerator: Accelerator instance.
        """
        # Always return raw scores for logging and (optionally) advantage weighting.
        reward_fn = multi_score(
            accelerator.device,
            cfg.reward_fn,
            cfg.reward_module,
            return_raw_scores=True,
        )
        self.reward_fn = reward_fn

        eval_reward_cfg = (
            cfg.eval_reward_fn if cfg.eval_reward_fn is not None else cfg.reward_fn
        )
        eval_reward_module = (
            cfg.eval_reward_module
            if cfg.eval_reward_module is not None
            else cfg.reward_module
        )
        eval_reward_fn = multi_score(
            accelerator.device,
            eval_reward_cfg,
            eval_reward_module,
            return_raw_scores=True,
        )
        self.eval_reward_fn = eval_reward_fn

    def _setup_dataloaders(self, cfg: Config, accelerator: Accelerator):
        """Setup data loaders and negative prompt embeddings.

        Args:
            cfg: Training config.
            accelerator: Accelerator instance.
        """
        train_dataloader, test_dataloader, train_sampler = build_dataloaders(
            cfg, accelerator
        )
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.train_sampler = train_sampler

        # Setup negative prompt embeddings
        neg_prompt_embed = wan_compute_text_embeddings(
            [""],
            self.text_encoders,
            self.tokenizers,
            max_sequence_length=512,
            device=accelerator.device,
        )
        sample_neg_prompt_embeds = neg_prompt_embed.repeat(cfg.sample.batch_size, 1, 1)
        train_neg_prompt_embeds = neg_prompt_embed.repeat(cfg.train.batch_size, 1, 1)
        self.sample_neg_prompt_embeds = sample_neg_prompt_embeds
        self.train_neg_prompt_embeds = train_neg_prompt_embeds

    def _setup_stat_trackers(self, cfg: Config):
        """Setup stat trackers for advantage computation.

        Args:
            cfg: Training config.
        """
        if cfg.sample.num_video_per_prompt == 1:
            cfg.per_prompt_stat_tracking = False

        if cfg.per_prompt_stat_tracking:
            stat_tracker = PerPromptStatTracker(
                use_global_std=cfg.sample.global_std,
                max_group_std=cfg.sample.max_group_std,
            )
            self.stat_tracker = stat_tracker
        else:
            self.stat_tracker = None

        # Initialize reward_stat_trackers for mode 2 (weight_advantages=True)
        reward_stat_trackers = None
        kl_stat_tracker = None
        if cfg.train.weight_advantages and cfg.per_prompt_stat_tracking:
            reward_stat_trackers = {
                reward_name: PerPromptStatTracker(
                    use_global_std=cfg.sample.global_std,
                    max_group_std=cfg.sample.max_group_std,
                )
                for reward_name in cfg.reward_fn.keys()
            }
            self.reward_stat_trackers = reward_stat_trackers
            # Initialize KL stat tracker if KL reward is enabled
            if cfg.sample.kl_reward > 0:
                kl_stat_tracker = PerPromptStatTracker(
                    use_global_std=cfg.sample.global_std,
                    max_group_std=cfg.sample.max_group_std,
                )
                self.kl_stat_tracker = kl_stat_tracker
            else:
                self.kl_stat_tracker = None
        else:
            self.reward_stat_trackers = None
            self.kl_stat_tracker = None

    def _setup_training_utilities(self, cfg: Config, accelerator: Accelerator):
        """Setup training utilities like autocast and executor.

        Args:
            cfg: Training config.
            accelerator: Accelerator instance.
        """
        autocast = contextlib.nullcontext if cfg.use_lora else accelerator.autocast
        self.autocast = autocast

        # async reward executor
        executor = futures.ThreadPoolExecutor(max_workers=8)
        self.executor = executor

    def _prepare_models_with_accelerator(self, cfg: Config, accelerator: Accelerator):
        """Prepare models with accelerator and setup EMA.

        Args:
            cfg: Training config.
            accelerator: Accelerator instance.
        """
        transformer, optimizer, test_dataloader = accelerator.prepare(
            self.transformer, self.optimizer, self.test_dataloader
        )
        self.transformer = transformer
        self.optimizer = optimizer
        self.test_dataloader = test_dataloader

        # Keep pipeline reference in sync with the wrapped (e.g., FSDP) model
        self.pipeline.transformer = transformer

        # Prepare ref_transformer if it exists
        if hasattr(self, "ref_transformer") and self.ref_transformer is not None:
            ref_transformer = accelerator.prepare_model(
                self.ref_transformer, evaluation_mode=True
            )
            ref_transformer.eval()
            self.pipeline.ref_transformer = ref_transformer
            self.ref_transformer = ref_transformer

        # Rebuild trainable params on wrapped model and init EMA after wrapping
        transformer_params = [p for p in transformer.parameters() if p.requires_grad]
        self.transformer_params = transformer_params

        ema = None
        if cfg.train.ema:
            ema = EMAModuleWrapper(
                transformer_params,
                decay=cfg.train.ema_decay,
                update_step_interval=cfg.train.ema_update_interval,
                device=accelerator.device,
            )
        self.ema = ema

        train_iter = iter(self.train_dataloader)
        self.train_iter = train_iter

    def _on_resume_from_checkpoint(self, accelerator: Accelerator):
        """WAN-specific resume logic: reload ref_transformer if needed.

        Args:
            accelerator: Accelerator instance.
        """
        cfg = self.cfg

        # Ensure ref_transformer exists after resume
        # Always reload from pretrained_model to ensure it uses original weights
        if self.full_finetune and cfg.train.beta > 0:
            if (
                self.ref_transformer is None
                or not hasattr(self.pipeline, "ref_transformer")
                or self.pipeline.ref_transformer is None
            ):
                if accelerator.is_main_process:
                    logger.info(
                        f"Loading ref_transformer from pretrained_model: {cfg.paths.pretrained_model}"
                    )
                # Reload ref_transformer from pretrained_model path
                with fast_init(accelerator.device, init_weights=False):
                    ref_pipeline = WanPipeline.from_pretrained(
                        cfg.paths.pretrained_model
                    )
                ref_transformer = ref_pipeline.transformer
                ref_transformer.requires_grad_(False)
                ref_transformer = accelerator.prepare_model(
                    ref_transformer, evaluation_mode=True
                )
                ref_transformer.eval()
                self.pipeline.ref_transformer = ref_transformer
                self.ref_transformer = ref_transformer

    def _run_training_loop(
        self, cfg: Config, accelerator: Accelerator, first_epoch: int, global_step: int
    ):
        """Run the main training loop.

        Args:
            cfg: Training config.
            accelerator: Accelerator instance.
            first_epoch: Starting epoch number.
            global_step: Starting global step number.
        """
        resume_path = self.resume_path

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

        if accelerator.is_main_process:
            logger.info(
                "\n".join(
                    [
                        "\n***** Running training *****",
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
            self.pipeline.transformer.eval()

            if (
                epoch % cfg.eval_freq == 0
                and (epoch > 0 or cfg.initial_eval)
                and not (resume_path and epoch == first_epoch)
            ):
                wan_eval_once(
                    cfg,
                    accelerator,
                    self.pipeline,
                    self.test_dataloader,
                    self.text_encoders,
                    self.tokenizers,
                    self.sample_neg_prompt_embeds,
                    self.eval_reward_fn,
                    self.autocast,
                    global_step,
                    self.ema,
                    self.transformer_params,
                    log_metrics=self.log_metrics,
                )
            # Per-epoch seeding for reproducible sampling (e.g., when generator=None / same_latent=False or calculate step-wise log_prob during sampling)
            set_seed(cfg.seed + epoch, device_specific=True)
            if (
                epoch % cfg.save_freq == 0
                and epoch > 0
                and not (
                    resume_path and epoch == first_epoch
                )  # don't save on the resume epoch
            ):
                current_epoch_tag = epoch * cfg.sample.num_batches_per_epoch
                self.save_checkpoint(
                    self.transformer,
                    global_step,
                    epoch,
                    current_epoch_tag,
                )

            samples = wan_sample_epoch(
                cfg,
                accelerator,
                self.pipeline,
                self.train_sampler,
                self.train_iter,
                self.reward_fn,
                self.sample_neg_prompt_embeds,
                self.text_encoders,
                self.tokenizers,
                self.executor,
                self.autocast,
                epoch,
                global_step,
            )
            # Prepare samples for training (collate, process rewards, compute advantages)
            samples = self._prepare_samples_for_training(samples, epoch, global_step)

            total_batch_size, num_timesteps = samples["timesteps"].shape

            for inner_epoch in range(cfg.train.num_inner_epochs):
                # Use deterministic generator for reproducibility
                # Seed based on epoch and inner_epoch to ensure consistency across runs
                generator = torch.Generator(device=accelerator.device)
                generator.manual_seed(
                    cfg.seed + epoch * SEED_EPOCH_STRIDE + inner_epoch
                )
                perm = torch.randperm(
                    total_batch_size, device=accelerator.device, generator=generator
                )
                samples = {k: v[perm] for k, v in samples.items()}
                perms = torch.stack(
                    [
                        torch.arange(num_timesteps, device=accelerator.device)
                        for _ in range(total_batch_size)
                    ]
                )
                for key in ["timesteps", "latents", "next_latents", "log_probs"]:
                    samples[key] = samples[key][
                        torch.arange(total_batch_size, device=accelerator.device)[
                            :, None
                        ],
                        perms,
                    ]

                micoe_batch = total_batch_size // cfg.sample.num_batches_per_epoch
                samples_batched = {
                    k: v.reshape(-1, micoe_batch, *v.shape[1:])
                    for k, v in samples.items()
                }
                samples_batched = [
                    dict(zip(samples_batched, x))
                    for x in zip(*samples_batched.values())
                ]

                self.pipeline.transformer.train()
                info = defaultdict(list)
                for i, sample in tqdm(
                    list(enumerate(samples_batched)),
                    desc=f"Epoch {epoch}.{inner_epoch}: training",
                    position=0,
                    disable=not accelerator.is_local_main_process,
                ):
                    if cfg.train.cfg:
                        embeds = sample["prompt_embeds"]
                        negative_embeds = self.train_neg_prompt_embeds[
                            : len(sample["prompt_embeds"])
                        ]
                    else:
                        embeds = sample["prompt_embeds"]
                        negative_embeds = None

                    for j in tqdm(
                        self.train_timesteps,
                        desc="Timestep",
                        position=1,
                        leave=False,
                        disable=not accelerator.is_local_main_process,
                    ):
                        # Compute reference model output BEFORE accumulate context
                        # This avoids checkpointing issues when model structure changes when using LoRA
                        prev_sample_ref = None
                        log_prob_ref = None
                        prev_sample_mean_ref = None
                        std_dev_t_ref = None
                        dt_sqrt_ref = None

                        if cfg.train.beta > 0:
                            if self.full_finetune:
                                ref_model = self.ref_transformer
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
                                        dt_sqrt_ref,
                                        sigma_ref,
                                        sigma_max_ref,
                                    ) = wan_compute_log_prob(
                                        ref_model,
                                        self.pipeline,
                                        sample,
                                        j,
                                        embeds,
                                        negative_embeds,
                                        cfg,
                                    )
                            else:
                                # LoRA case: compute reference output outside accumulate context
                                with torch.no_grad():
                                    with self.transformer.module.disable_adapter():
                                        (
                                            prev_sample_ref,
                                            log_prob_ref,
                                            prev_sample_mean_ref,
                                            std_dev_t_ref,
                                            dt_sqrt_ref,
                                            sigma_ref,
                                            sigma_max_ref,
                                        ) = wan_compute_log_prob(
                                            self.transformer,
                                            self.pipeline,
                                            sample,
                                            j,
                                            embeds,
                                            negative_embeds,
                                            cfg,
                                        )

                        # Main model forward and backward in accumulate context
                        # Model structure remains consistent (adapter always enabled)
                        with accelerator.accumulate(self.transformer):
                            with self.autocast():
                                (
                                    prev_sample,
                                    log_prob,
                                    prev_sample_mean,
                                    std_dev_t,
                                    dt_sqrt,
                                    sigma,
                                    sigma_max,
                                ) = wan_compute_log_prob(
                                    self.transformer,
                                    self.pipeline,
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

                            reweight_scale = 1.0
                            reweight_scale_kl = 1.0
                            if (
                                cfg.train.loss_reweighting == "longcat"
                                and cfg.sample.sde_type == "flow_sde"
                            ):
                                reweight_scale = (
                                    torch.sqrt(
                                        sigma
                                        / (
                                            1
                                            - torch.where(
                                                sigma == 1,
                                                torch.tensor(
                                                    sigma_max,
                                                    device=sigma.device,
                                                    dtype=sigma.dtype,
                                                ),
                                                sigma,
                                            )
                                        )
                                    )
                                    / dt_sqrt
                                )
                                reweight_scale_kl = reweight_scale**2

                            if cfg.train.beta > 0:
                                if cfg.sample.sde_type == "flow_sde":
                                    kl_denom = (std_dev_t * dt_sqrt_ref) ** 2
                                elif cfg.sample.sde_type == "flow_cps":
                                    kl_denom = 1 / 2
                                else:
                                    raise ValueError(
                                        f"Unknown sde_type: {cfg.sample.sde_type}. Must be 'flow_sde' or 'flow_cps'."
                                    )
                                kl_loss = (
                                    (prev_sample_mean - prev_sample_mean_ref) ** 2
                                ).mean(dim=(1, 2, 3), keepdim=True) / (2 * kl_denom)
                                kl_loss = torch.mean(kl_loss)
                                loss = (
                                    reweight_scale * policy_loss
                                    + cfg.train.beta * kl_loss * reweight_scale_kl
                                )
                            else:
                                loss = reweight_scale * policy_loss

                            info["approx_kl"].append(
                                0.5
                                * torch.mean(
                                    (log_prob - sample["log_probs"][:, j]) ** 2
                                )
                            )
                            info["clip_frac"].append(
                                torch.mean(
                                    (
                                        torch.abs(ratio - 1.0) > cfg.train.clip_range
                                    ).float()
                                )
                            )
                            info["clip_frac_gt_one"].append(
                                torch.mean((ratio - 1.0 > cfg.train.clip_range).float())
                            )
                            info["clip_frac_lt_one"].append(
                                torch.mean((1.0 - ratio > cfg.train.clip_range).float())
                            )
                            info["policy_loss"].append(policy_loss)
                            if cfg.train.beta > 0:
                                info["kl_loss"].append(kl_loss)
                            info["loss"].append(loss)

                            accelerator.backward(loss)
                            if accelerator.sync_gradients:
                                accelerator.clip_grad_norm_(
                                    self.transformer.parameters(),
                                    cfg.train.max_grad_norm,
                                )
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                        if accelerator.sync_gradients:
                            info = {
                                k: torch.mean(torch.stack(v)) for k, v in info.items()
                            }
                            info = accelerator.reduce(info, reduction="mean")
                            info.update({"epoch": epoch, "inner_epoch": inner_epoch})
                            # Log to trackers and to stdout (scalar-only), with caller context
                            self.log_metrics(accelerator, info, global_step)
                            global_step += 1
                            info = defaultdict(list)
                    if cfg.train.ema:
                        self.ema.step(self.transformer_params, global_step)

        # Save final model after training completes
        if accelerator.is_main_process:
            logger.info("Training completed. Saving final model...")
        accelerator.wait_for_everyone()

        # Apply EMA weights to model before saving (on all processes, before unwrap)
        if cfg.train.ema:
            self.ema.copy_ema_to(self.transformer_params, store_temp=True)

        base_transformer = unwrap_model(self.transformer, accelerator)

        # Prepare directories (only on main process)
        if accelerator.is_main_process:
            final_model_dir = os.path.join(cfg.paths.save_dir, "final_model")
            os.makedirs(final_model_dir, exist_ok=True)
        else:
            final_model_dir = None

        # Critical: save_pretrained may call model.state_dict() internally:
        # - For LoRA: PeftModel.save_pretrained -> get_peft_model_state_dict -> model.state_dict()
        # - For full finetune: PreTrainedModel.save_pretrained -> model.state_dict()
        # Get state_dict on ALL processes first (ensures all participate in unshard if needed)
        # Then pass it to save_pretrained to avoid calling model.state_dict() again
        state_dict_to_save = base_transformer.state_dict()

        # Save model - use try-finally to ensure cleanup even if save fails
        try:
            if accelerator.is_main_process:
                # Pass state_dict explicitly to avoid save_pretrained calling model.state_dict() again
                base_transformer.save_pretrained(
                    final_model_dir, state_dict=state_dict_to_save
                )
                logger.info(f"Final model saved to {final_model_dir}")
        finally:
            if cfg.train.ema:
                self.ema.copy_temp_to(self.transformer_params)

            # Synchronize after saving to ensure all processes complete
            accelerator.wait_for_everyone()

    def _prepare_samples_for_training(
        self, samples: List[Dict[str, Any]], epoch: int, global_step: int
    ) -> Dict[str, torch.Tensor]:
        """Prepare samples for training by collating, processing rewards, and computing advantages.

        Args:
            samples: List of sample dicts from sampling epoch.
            epoch: Current epoch.
            global_step: Current global step.

        Returns:
            Processed samples dict ready for training.
        """
        cfg = self.cfg
        accelerator = self.accelerator

        # Collate list of per-batch samples into dict of tensors
        samples = {
            k: (
                torch.cat([s[k] for s in samples], dim=0)
                if not isinstance(samples[0][k], dict)
                else {
                    sub_key: torch.cat([s[k][sub_key] for s in samples], dim=0)
                    for sub_key in samples[0][k]
                }
            )
            for k in samples[0].keys()
        }

        samples["rewards"]["ori_avg"] = samples["rewards"]["avg"]
        # Apply KL penalty only to avg (raw scores remain unchanged)
        kl_penalty = (
            cfg.sample.kl_reward * samples["kl"]
        )  # Shape: (batch_size, num_timesteps)
        samples["rewards"]["avg"] = samples["rewards"]["avg"].unsqueeze(-1) - kl_penalty

        # Save original raw rewards and broadcast them to (batch_size, num_timesteps)
        num_timesteps = samples["kl"].shape[1]
        for reward_name in cfg.reward_fn.keys():
            raw_reward_key = f"{reward_name}_raw"
            # Save original raw reward
            samples["rewards"][f"ori_{raw_reward_key}"] = samples["rewards"][
                raw_reward_key
            ]
            # Broadcast raw reward from (batch_size,) to (batch_size, num_timesteps)
            samples["rewards"][raw_reward_key] = (
                samples["rewards"][raw_reward_key]
                .unsqueeze(-1)
                .expand(-1, num_timesteps)
            )

        gathered_rewards = {
            key: accelerator.gather(value) for key, value in samples["rewards"].items()
        }
        gathered_rewards = {
            key: value.cpu().numpy() for key, value in gathered_rewards.items()
        }
        # Gather KL for advantage calculation in weight_advantages mode
        gathered_kl = (
            accelerator.gather(samples["kl"]).cpu().numpy()
        )  # Shape: (total_batch_size, num_timesteps)

        # log rewards and KL - only log raw scores
        raw_keys = [
            k
            for k in gathered_rewards.keys()
            if k.endswith("_raw") and not k.startswith("ori_")
        ]
        reward_logs = {
            f"reward_{key}": value.mean()
            for key, value in gathered_rewards.items()
            if key in raw_keys
        }
        kl_mean = samples["kl"].mean().detach().cpu().item()
        kl_abs = samples["kl"].abs().mean().detach().cpu().item()
        self.log_metrics(
            accelerator,
            {
                "epoch": epoch,
                **reward_logs,
                "kl": kl_mean,
                "kl_abs": kl_abs,
            },
            global_step,
        )

        # Compute advantages using the abstracted function
        advantages, advantage_log_dict = compute_advantages(
            cfg=cfg,
            accelerator=accelerator,
            pipeline=self.pipeline,
            samples=samples,
            gathered_rewards=gathered_rewards,
            gathered_kl=gathered_kl,
            stat_tracker=self.stat_tracker if cfg.per_prompt_stat_tracking else None,
            reward_stat_trackers=(
                self.reward_stat_trackers
                if cfg.train.weight_advantages and cfg.per_prompt_stat_tracking
                else None
            ),
            kl_stat_tracker=(
                self.kl_stat_tracker
                if cfg.train.weight_advantages
                and cfg.per_prompt_stat_tracking
                and cfg.sample.kl_reward > 0
                else None
            ),
        )

        # Log advantage-related metrics
        if advantage_log_dict:
            self.log_metrics(accelerator, advantage_log_dict, global_step)

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
            samples["advantages"] = samples["advantages"] + ADVANTAGE_EPSILON
            mask = samples["advantages"].abs().sum(dim=1) != 0
        if true_count % num_batches != 0:
            false_indices = torch.where(~mask)[0]
            num_to_change = num_batches - (true_count % num_batches)
            if len(false_indices) >= num_to_change:
                # Use deterministic generator for reproducibility
                # Seed based on epoch to ensure consistency across runs
                generator = torch.Generator(device=accelerator.device)
                generator.manual_seed(cfg.seed + epoch)
                random_indices = torch.randperm(
                    len(false_indices), device=accelerator.device, generator=generator
                )[:num_to_change]
                mask[false_indices[random_indices]] = True
        self.log_metrics(
            accelerator,
            {
                "actual_batch_size": mask.sum().item() // num_batches,
            },
            global_step,
        )
        samples = {k: v[mask] for k, v in samples.items()}

        return samples
