"""Base trainer class with common utilities for all trainers."""

import os
import datetime
import time
import json
import re
import shutil
import warnings
import inspect
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple
import torch
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from loguru import logger
from diffusers.utils.torch_utils import is_compiled_module

from video_grpo.config import Config
from video_grpo.exceptions import ConfigurationError
from video_grpo.utils import (
    build_accelerator,
    resolve_resume_checkpoint,
    fast_init,
    unwrap_model,
)


class BaseTrainer(ABC):
    """Base class for all trainers with common initialization and utilities.

    This is an abstract base class. Subclasses must implement the `train()` method.
    This class provides common functionality that can be reused across different
    trainer implementations. Subclasses should implement trainer-specific logic.
    """

    def __init__(self, cfg: Config):
        """Initialize base trainer with common setup.

        Args:
            cfg: Training configuration.

        Raises:
            ConfigurationError: If configuration is invalid.
        """
        self.cfg = cfg
        self._validate_config()
        self._setup_paths()
        self.accelerator = None
        self.pipeline = None
        self.optimizer = None
        self.train_dataloader = None
        self.test_dataloader = None
        self.ema = None
        self.transformer_params = None

    def _validate_config(self):
        """Validate training configuration.

        Raises:
            ConfigurationError: If configuration is invalid.
        """
        errors = []

        if not hasattr(self.cfg, "reward_fn") or not self.cfg.reward_fn:
            errors.append("reward_fn cannot be empty")

        if (
            hasattr(self.cfg.train, "weight_advantages")
            and self.cfg.train.weight_advantages
        ):
            if not self.cfg.reward_fn:
                errors.append(
                    "weight_advantages=True requires at least one reward in reward_fn"
                )

        # Warn if sde_window_size is 0 but sde_window_range is set
        # Also validate and truncate sde_window_range to [0, num_steps]
        if hasattr(self.cfg.sample, "sde_window_size") and hasattr(
            self.cfg.sample, "sde_window_range"
        ):
            sde_window_size = getattr(self.cfg.sample, "sde_window_size", 0) or 0
            sde_window_range = getattr(self.cfg.sample, "sde_window_range", None)
            if sde_window_size == 0 and sde_window_range is not None:
                warnings.warn(
                    "sde_window_size is 0, so sde_window_range will not be used. "
                    "Set sde_window_size > 0 to enable window-based training.",
                    UserWarning,
                    stacklevel=2,
                )

            # Validate and truncate sde_window_range to [0, num_steps]
            if sde_window_range is not None and hasattr(self.cfg.sample, "num_steps"):
                num_steps = self.cfg.sample.num_steps
                original_range = tuple(sde_window_range)
                new_start = max(0, min(sde_window_range[0], num_steps))
                new_end = max(0, min(sde_window_range[1], num_steps))

                # Ensure start <= end
                if new_start > new_end:
                    new_start = new_end

                truncated_range = (new_start, new_end)

                if truncated_range != original_range:
                    # Update the config with truncated range
                    self.cfg.sample.sde_window_range = truncated_range
                    warnings.warn(
                        f"sde_window_range {original_range} was truncated to "
                        f"{truncated_range} to fit within [0, {num_steps}] "
                        f"(num_steps).",
                        UserWarning,
                        stacklevel=2,
                    )

                # Check if truncated range is large enough for sde_window_size
                if sde_window_size > 0:
                    range_span = truncated_range[1] - truncated_range[0]
                    if range_span < sde_window_size:
                        errors.append(
                            f"sde_window_range {truncated_range} has span {range_span}, "
                            f"which is less than sde_window_size {sde_window_size}. "
                            f"Range span must be >= sde_window_size."
                        )

        if errors:
            raise ConfigurationError(f"Configuration errors: {', '.join(errors)}")

    def _setup_paths(self):
        """Setup run paths and directories.

        In distributed training, all processes must use the same timestamp to create
        the same directory. This method ensures that:
        1. Only the main process (LOCAL_RANK=0 or not in distributed mode) generates the timestamp
        2. Other processes wait for and use the same timestamp via file-based synchronization
        3. The timestamp is written to a lock file in the base save_dir directory
        """
        # Check if we're in distributed training by checking for RANK or LOCAL_RANK env var
        # RANK is global rank across all machines (0 for main process in multi-machine training)
        # LOCAL_RANK is local rank within a machine (0 for main process on each machine)
        # Prefer RANK for multi-machine training, fallback to LOCAL_RANK for single-machine
        rank = os.environ.get("RANK")
        local_rank = os.environ.get("LOCAL_RANK")
        # Main process is RANK=0 (global) or LOCAL_RANK=0 (single machine) or not in distributed mode
        is_main_process = (rank is None and local_rank is None) or rank == "0" or (rank is None and local_rank == "0")

        # Base save directory (before adding run_name)
        base_save_dir = self.cfg.paths.save_dir

        # Create a temporary lock file path for timestamp synchronization.
        # Prefer a run-specific suffix when available to avoid stale reuse.
        run_id = (
            os.environ.get("TORCHELASTIC_RUN_ID")
            or os.environ.get("RDZV_ID")
            or os.environ.get("SLURM_JOB_ID")
            or os.environ.get("LSB_JOBID")
            or os.environ.get("PBS_JOBID")
            or os.environ.get("JOB_ID")
        )
        lock_suffix = f".{run_id}" if run_id else ""
        timestamp_lock_file = os.path.join(
            base_save_dir, f".run_timestamp_lock{lock_suffix}"
        )

        if is_main_process:
            # Main process generates the timestamp
            unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")

            # Ensure base directory exists
            os.makedirs(base_save_dir, exist_ok=True)

            # Write timestamp to lock file atomically
            temp_file = timestamp_lock_file + ".tmp"
            with open(temp_file, "w") as f:
                f.write(unique_id)
            os.rename(temp_file, timestamp_lock_file)  # Atomic rename on most filesystems
        else:
            # Non-main processes wait for and read the timestamp from lock file
            max_wait = 60  # Wait up to 60 seconds for main process to create the file
            max_stale_seconds = 300  # Ignore stale lock files from prior runs
            wait_time = 0
            while wait_time < max_wait:
                if os.path.exists(timestamp_lock_file):
                    try:
                        age_seconds = time.time() - os.path.getmtime(timestamp_lock_file)
                    except OSError:
                        age_seconds = max_stale_seconds + 1
                    if age_seconds <= max_stale_seconds:
                        break
                time.sleep(0.1)
                wait_time += 0.1

            if not os.path.exists(timestamp_lock_file):
                raise RuntimeError(
                    f"Timestamp lock file not created by main process after {max_wait}s. "
                    f"Expected at: {timestamp_lock_file}. "
                    "This may indicate a synchronization issue in distributed training."
                )

            # Double-check file is not stale before reading (defensive check)
            try:
                age_seconds = time.time() - os.path.getmtime(timestamp_lock_file)
                if age_seconds > max_stale_seconds:
                    raise RuntimeError(
                        f"Timestamp lock file exists but is stale (age={age_seconds:.1f}s > "
                        f"{max_stale_seconds}s). File: {timestamp_lock_file}"
                    )
            except OSError as e:
                raise RuntimeError(
                    f"Failed to check timestamp lock file age: {e}. "
                    f"File: {timestamp_lock_file}"
                ) from e

            # Read timestamp from lock file
            try:
                with open(timestamp_lock_file, "r") as f:
                    unique_id = f.read().strip()
                if not unique_id:
                    raise RuntimeError(
                        f"Timestamp lock file is empty: {timestamp_lock_file}"
                    )
            except (OSError, IOError) as e:
                raise RuntimeError(
                    f"Failed to read timestamp from lock file: {e}. "
                    f"File: {timestamp_lock_file}"
                ) from e

        self.cfg.run_name = (
            self.cfg.run_name + "_" + unique_id if self.cfg.run_name else unique_id
        )
        # Add run_name (with timestamp) as subdirectory to save_dir
        self.cfg.paths.save_dir = os.path.join(
            self.cfg.paths.save_dir, self.cfg.run_name
        )
        self.resume_path = resolve_resume_checkpoint(
            getattr(self.cfg.paths, "resume_from", None)
        )

    def log_metrics(
        self,
        accelerator: Accelerator,
        metrics: dict[str, Any],
        step: int,
    ) -> None:
        """Log metrics to trackers and (optionally) to stdout.

        Args:
            accelerator: The active :class:`Accelerator` instance.
            metrics: Dictionary of metric name to value. All entries are forwarded
                to ``accelerator.log``; only scalar-like values are printed.
            step: Global step to use for logging.
        """
        # 1) Always log to the configured tracker(s)
        accelerator.log(metrics, step=step)

        # 2) Only main process prints to stdout
        if not accelerator.is_main_process:
            return

        # Filter to scalar-like metrics only
        scalar_items: dict[str, float] = {}
        for key, value in metrics.items():
            # Skip None values
            if value is None:
                continue
            try:
                # Handle torch.Tensor first (before numpy check, as Tensor also has .item() and .dtype)
                if isinstance(value, torch.Tensor):
                    # Only log 0-dim tensors as scalars
                    if value.ndim == 0:
                        # Ensure tensor is on CPU for .item() call
                        if value.is_cuda:
                            value = value.cpu()
                        try:
                            scalar_items[key] = float(value.item())
                        except (ValueError, RuntimeError):
                            # Skip if .item() fails
                            continue
                # Handle Python native types (int, float)
                # Note: numpy.float64 is a subclass of float, but has dtype attribute
                # numpy.int64 is NOT a subclass of int, and has dtype attribute
                elif isinstance(value, (int, float)) and not hasattr(value, "dtype"):
                    scalar_items[key] = float(value)
                # Handle numpy scalars (numpy.int64, numpy.float64, etc.)
                # numpy scalars have .item() method and .dtype attribute
                elif hasattr(value, "item") and hasattr(value, "dtype"):
                    try:
                        scalar_items[key] = float(value.item())
                    except (AttributeError, ValueError, TypeError):
                        # Not a scalar numpy type, skip
                        continue
                # Skip non-scalar / non-numeric types (e.g., images, videos, dicts, lists)
            except Exception:
                # Skip any value that causes an exception during processing
                continue

        if not scalar_items:
            return

        # Build a compact "k=v" string
        parts = []
        for k, v in scalar_items.items():
            # Use 4 decimal places for floats, plain for ints
            if float(v).is_integer():
                parts.append(f"{k}={int(v)}")
            else:
                parts.append(f"{k}={v:.4f}")

        msg = " ".join(parts)

        # Use loguru's opt(depth=2) to show caller's location instead of this function's location.
        # depth=2: skip logger.info() call and log_metrics() function, show actual caller.
        # The loguru formatter will print file:line function() for the true caller.
        logger.opt(depth=2).info(f"[step {step}] | {msg}")

    def setup_accelerator(self, gradient_accumulation_steps: int) -> Accelerator:
        """Setup Accelerate accelerator.

        Args:
            gradient_accumulation_steps: Number of gradient accumulation steps.

        Returns:
            Configured Accelerator instance.
        """
        accelerator_config = ProjectConfiguration(
            project_dir=self.cfg.paths.save_dir,
            automatic_checkpoint_naming=False,  # Disable automatic checkpointing
        )
        self.accelerator = build_accelerator(
            self.cfg, gradient_accumulation_steps, accelerator_config
        )

        # Global seeding
        set_seed(self.cfg.seed, device_specific=True)

        # Global backend settings (configurable via Config)
        # cuDNN determinism / benchmarking
        torch.backends.cudnn.deterministic = self.cfg.cudnn_deterministic
        torch.backends.cudnn.benchmark = self.cfg.cudnn_benchmark
        # TF32 matmul on Tensor Cores
        if hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = self.cfg.allow_tf32

        if self.accelerator.is_main_process:
            self.accelerator.init_trackers(
                project_name=self.cfg.project_name,
                config=self.cfg.__dict__,
                init_kwargs={"wandb": {"name": self.cfg.run_name}},
            )
            logger.info(f"\n{self.cfg}")

        return self.accelerator

    def setup_pipeline(self, pipeline_class: Any, pretrained_path: str) -> Any:
        """Setup diffusion pipeline.

        Args:
            pipeline_class: Pipeline class to instantiate.
            pretrained_path: Path to pretrained model.

        Returns:
            Configured pipeline instance.
        """
        with fast_init(self.accelerator.device, init_weights=False):
            self.pipeline = pipeline_class.from_pretrained(pretrained_path)
        return self.pipeline

    def setup_mixed_precision_dtype(self) -> torch.dtype:
        """Determine inference dtype based on accelerator mixed precision.

        Returns:
            Appropriate dtype for inference.
        """
        inference_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            inference_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            inference_dtype = torch.bfloat16
        return inference_dtype

    def calculate_gradient_accumulation_steps(
        self, num_train_timesteps: int
    ) -> Tuple[int, int]:
        """Calculate gradient accumulation steps.

        Args:
            num_train_timesteps: Number of training timesteps.

        Returns:
            Tuple of (base_gas, total_gradient_accumulation_steps).
        """
        base_gas = self.cfg.train.gradient_accumulation_steps
        if base_gas is None or base_gas <= 0:
            total_chunks = self.cfg.sample.num_batches_per_epoch
            base_gas = total_chunks // 2 if total_chunks > 1 else 1
        self.cfg.train.gradient_accumulation_steps = base_gas
        gradient_accumulation_steps = base_gas * num_train_timesteps
        return base_gas, gradient_accumulation_steps

    def get_train_timesteps(self, num_train_timesteps: int) -> List[int]:
        """Get list of training timesteps.

        Args:
            num_train_timesteps: Number of training timesteps.

        Returns:
            List of timestep indices.
        """
        return [step_index for step_index in range(num_train_timesteps)]

    def setup_optimizer(
        self, parameters: List[torch.nn.Parameter], use_8bit: bool = False
    ) -> torch.optim.Optimizer:
        """Setup optimizer for training.

        Args:
            parameters: List of parameters to optimize.
            use_8bit: Whether to use 8-bit optimizer.

        Returns:
            Configured optimizer.
        """
        if use_8bit:
            import bitsandbytes as bnb

            optimizer_cls = bnb.optim.AdamW8bit
        else:
            optimizer_cls = torch.optim.AdamW

        self.optimizer = optimizer_cls(
            parameters,
            lr=self.cfg.train.learning_rate,
            betas=(self.cfg.train.adam_beta1, self.cfg.train.adam_beta2),
            weight_decay=self.cfg.train.adam_weight_decay,
            eps=self.cfg.train.adam_epsilon,
        )
        return self.optimizer

    def save_checkpoint(
        self,
        transformer: torch.nn.Module,
        global_step: int,
        epoch: int,
        current_epoch_tag: int,
    ) -> None:
        """Save state, EMA, metadata, and unwrapped model checkpoint.

        Args:
            transformer: Potentially wrapped transformer being trained.
            global_step: Current global step.
            epoch: Current epoch.
            current_epoch_tag: Sampler epoch_tag for resume alignment.
        """
        cfg = self.cfg
        accelerator = self.accelerator

        # save_dir already contains run_name, so checkpoints will be saved in run directory
        checkpoints_dir = os.path.join(cfg.paths.save_dir, "checkpoints")
        save_root = os.path.join(checkpoints_dir, f"checkpoint-{global_step}")

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
        if cfg.train.ema and self.ema is not None:
            ema_dir = os.path.join(save_root, "ema")
            if accelerator.is_main_process:
                os.makedirs(ema_dir, exist_ok=True)
            accelerator.wait_for_everyone()
            ema_path = os.path.join(
                ema_dir, f"ema_state_rank{accelerator.process_index}.pt"
            )
            torch.save(self.ema.state_dict(), ema_path)

        # Apply EMA weights to model before saving (on all processes, before unwrap)
        if cfg.train.ema and self.ema is not None:
            self.ema.copy_ema_to(self.transformer_params, store_temp=True)

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
            if cfg.train.ema and self.ema is not None:
                self.ema.copy_temp_to(self.transformer_params)

        # Clean up old checkpoints after successful save (only on main process)
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
                    if os.path.isdir(checkpoint_path) and item.startswith(
                        "checkpoint-"
                    ):
                        # Extract step number from folder name (e.g., "checkpoint-120" -> 120)
                        match = re.search(r"checkpoint-(\d+)", item)
                        if match:
                            step_num = int(match.group(1))
                            checkpoint_folders.append((step_num, checkpoint_path))

                # Sort by step number (oldest first)
                checkpoint_folders.sort(key=lambda x: x[0])

                # Delete oldest checkpoints if we exceed the limit
                # After save, we now have len(checkpoint_folders) checkpoints total
                if len(checkpoint_folders) > cfg.num_checkpoint_limit:
                    num_to_delete = len(checkpoint_folders) - cfg.num_checkpoint_limit
                    for step_num, folder_path in checkpoint_folders[:num_to_delete]:
                        shutil.rmtree(folder_path)
                        logger.info(f"Deleted old checkpoint: checkpoint-{step_num}")

        # Final synchronization: ensure all processes complete all save operations before returning
        accelerator.wait_for_everyone()

    def resume_from_checkpoint(
        self, accelerator: Accelerator
    ) -> tuple[int, int, int | None]:
        """Resume training from checkpoint.

        Args:
            accelerator: Accelerator instance.

        Returns:
            Tuple of (first_epoch, global_step, resume_epoch_tag).
        """
        cfg = self.cfg
        resume_path = self.resume_path
        first_epoch = 0
        global_step = 0
        resume_epoch_tag = None

        if resume_path:
            if accelerator.is_main_process:
                logger.info(f"Resuming from {resume_path}")
            accelerator.load_state(resume_path)

            meta_path = os.path.join(resume_path, "metadata.json")
            if os.path.exists(meta_path):
                with open(meta_path, "r") as f:
                    metadata = json.load(f)
                global_step = metadata.get("global_step", 0)
                first_epoch = metadata.get("epoch", 0)
                resume_epoch_tag = metadata.get("current_epoch_tag", None)

            if cfg.train.ema and self.ema is not None:
                # Load per-rank EMA state; required for FSDP shard alignment.
                ema_state_path = os.path.join(
                    resume_path,
                    "ema",
                    f"ema_state_rank{accelerator.process_index}.pt",
                )
                if os.path.exists(ema_state_path):
                    ema_state = torch.load(
                        ema_state_path, map_location=accelerator.device
                    )
                    self.ema.load_state_dict(ema_state)
                    self.ema.to(accelerator.device)
                else:
                    raise ValueError(f"No EMA state found at {ema_state_path}")

            # Call hook method for trainer-specific resume logic
            self._on_resume_from_checkpoint(accelerator)

        return first_epoch, global_step, resume_epoch_tag

    def _on_resume_from_checkpoint(self, accelerator: Accelerator):
        """Hook method called after loading checkpoint state.

        Subclasses can override this to perform trainer-specific resume operations,
        such as reloading reference models or other trainer-specific components.

        Args:
            accelerator: Accelerator instance.
        """
        pass

    @abstractmethod
    def train(self):
        """Main training loop. Must be implemented by subclasses.

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError("Subclasses must implement train() method")
