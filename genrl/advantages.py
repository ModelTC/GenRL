"""Advantage computation utilities for GRPO training."""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from accelerate import Accelerator
from loguru import logger

from genrl.config import Config
from genrl.constants import EPSILON
from genrl.exceptions import ConfigurationError
from genrl.stat_tracking import PerPromptStatTracker
from genrl.utils import calculate_zero_std_ratio


def _normalize_rewards(rewards: np.ndarray, epsilon: float = EPSILON) -> np.ndarray:
    """Normalize rewards to zero mean and unit variance.

    Args:
        rewards: Reward array of shape (batch_size, ...).
        epsilon: Small value for numerical stability.

    Returns:
        Normalized rewards with same shape as input.
    """
    return (rewards - rewards.mean()) / (rewards.std() + epsilon)


def _compute_kl_advantages(
    gathered_kl: np.ndarray,
    kl_stat_tracker: Optional[PerPromptStatTracker],
    prompts: Optional[List[str]],
    use_per_prompt: bool,
) -> np.ndarray:
    """Compute KL advantages (negative because KL is a penalty).

    Args:
        gathered_kl: KL divergence array, shape (total_batch_size, num_timesteps).
        kl_stat_tracker: Per-prompt stat tracker if enabled.
        prompts: List of prompts for per-prompt tracking.
        use_per_prompt: Whether to use per-prompt tracking.

    Returns:
        KL advantages (negative), shape (total_batch_size, num_timesteps).
    """
    if use_per_prompt and kl_stat_tracker is not None:
        # KL is a penalty (larger KL is worse), so use negative KL
        return kl_stat_tracker.update(prompts, -gathered_kl)
    else:
        # Direct normalization on full shape
        # Normalize negative KL to maintain consistency with per_prompt mode
        return _normalize_rewards(-gathered_kl)


def compute_advantages(  # noqa: PLR0913, PLR0912, PLR0915
    cfg: Config,
    accelerator: Accelerator,
    pipeline: Any,  # Any pipeline with tokenizer.batch_decode method (e.g., diffusers.DiffusionPipeline)
    samples: Dict[str, Any],
    gathered_rewards: Dict[str, np.ndarray],
    gathered_kl: np.ndarray,
    stat_tracker: Optional[PerPromptStatTracker],
    reward_stat_trackers: Optional[Dict[str, PerPromptStatTracker]],
    kl_stat_tracker: Optional[PerPromptStatTracker],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Compute advantages from gathered rewards and KL divergence.

    Supports two modes:
    - Mode 1 (default): Weight rewards first, then compute advantages
    - Mode 2 (weight_advantages=True): Compute advantages for each reward separately, then weight them

    Args:
        cfg: Training config.
        accelerator: Accelerator for distributed operations.
        pipeline: Pipeline with tokenizer for prompt decoding (must have tokenizer.batch_decode method).
        samples: Sample dict containing prompt_ids.
        gathered_rewards: Dict of gathered reward arrays (numpy).
        gathered_kl: Gathered KL divergence array (numpy), shape (total_batch_size, num_timesteps).
        stat_tracker: Per-prompt stat tracker for Mode 1 (if per_prompt_stat_tracking enabled).
        reward_stat_trackers: Dict of per-prompt stat trackers for Mode 2 (if enabled).
        kl_stat_tracker: Per-prompt stat tracker for KL in Mode 2 (if enabled).

    Returns:
        Tuple of (advantages array, log_dict for accelerator.log).
    """
    log_dict = {}

    if cfg.train.weight_advantages:
        # Mode 2: Compute advantages for each reward separately, then weight them
        if cfg.per_prompt_stat_tracking:
            if reward_stat_trackers is None:
                msg = (
                    "reward_stat_trackers must be provided when weight_advantages=True "
                    "and per_prompt_stat_tracking=True"
                )
                raise ConfigurationError(msg)
            prompt_ids = accelerator.gather(samples["prompt_ids"]).cpu().numpy()
            prompts = pipeline.tokenizer.batch_decode(
                prompt_ids, skip_special_tokens=True
            )

            # Compute advantages for each raw reward separately
            weighted_advantages_list = []

            for reward_name in cfg.reward_fn:
                raw_reward_key = f"{reward_name}_raw"
                # Compute advantage for this reward using its own stat_tracker
                reward_advantages = reward_stat_trackers[reward_name].update(
                    prompts, gathered_rewards[raw_reward_key]
                )  # Shape: (total_batch_size, num_timesteps)
                # Weight the advantages
                weight = cfg.reward_fn[reward_name]
                weighted_advantages_list.append(reward_advantages * weight)

            # Handle KL as a reward: compute advantage for KL, then subtract with kl_reward weight
            if cfg.sample.kl_reward > 0:
                if kl_stat_tracker is None:
                    msg = (
                        "kl_stat_tracker must be provided when weight_advantages=True, "
                        "per_prompt_stat_tracking=True, and kl_reward > 0"
                    )
                    raise ConfigurationError(msg)
                kl_advantages = _compute_kl_advantages(
                    gathered_kl, kl_stat_tracker, prompts, use_per_prompt=True
                )
                # Subtract KL advantages with kl_reward as weight
                # kl_advantages is already negative (because KL is a penalty),
                # so we directly multiply by kl_reward to get a negative contribution
                weighted_advantages_list.append(kl_advantages * cfg.sample.kl_reward)

            # Sum weighted advantages
            advantages = sum(weighted_advantages_list)

            if accelerator.is_local_main_process:
                logger.info(
                    f"len(prompts) {len(prompts)} | len unique {len(set(prompts))}"
                )
            # Use the first stat_tracker for logging
            first_reward_name = next(iter(cfg.reward_fn))
            group_size, trained_prompt_num = reward_stat_trackers[
                first_reward_name
            ].get_stats()
            # Calculate zero_std_ratio for each raw reward
            zero_std_ratios = {}
            for reward_name in cfg.reward_fn:
                raw_reward_key = f"{reward_name}_raw"
                zero_std_ratios[f"zero_std_ratio_{reward_name}"] = (
                    calculate_zero_std_ratio(
                        prompts, gathered_rewards, reward_key=f"ori_{raw_reward_key}"
                    )
                )
            log_dict = {
                "group_size": group_size,
                "trained_prompt_num": trained_prompt_num,
                **zero_std_ratios,
            }
            # Clear all reward stat_trackers and KL stat tracker
            if reward_stat_trackers is not None:
                for tracker in reward_stat_trackers.values():
                    tracker.clear()
            if kl_stat_tracker is not None:
                kl_stat_tracker.clear()
        else:
            # No per-prompt tracking: compute advantages for each raw reward, then weight
            weighted_advantages_list = []
            for reward_name in cfg.reward_fn:
                raw_reward_key = f"{reward_name}_raw"
                raw_rewards = gathered_rewards[
                    raw_reward_key
                ]  # Shape: (total_batch_size, num_timesteps)
                # Direct normalization on full shape
                reward_advantages = _normalize_rewards(raw_rewards)
                # Weight the advantages
                weight = cfg.reward_fn[reward_name]
                weighted_advantages_list.append(reward_advantages * weight)

            # Handle KL as a reward: compute advantage for KL, then subtract with kl_reward weight
            if cfg.sample.kl_reward > 0:
                kl_advantages = _compute_kl_advantages(
                    gathered_kl, None, None, use_per_prompt=False
                )
                # Subtract KL advantages with kl_reward as weight
                # kl_advantages is already negative (because KL is a penalty),
                # so we directly multiply by kl_reward to get a negative contribution
                weighted_advantages_list.append(kl_advantages * cfg.sample.kl_reward)

            # Sum weighted advantages
            advantages = sum(weighted_advantages_list)
    else:
        # Mode 1 (default): Weight rewards first, then compute advantages
        if cfg.per_prompt_stat_tracking:
            if stat_tracker is None:
                raise ConfigurationError(
                    "stat_tracker must be provided when per_prompt_stat_tracking=True"
                )
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
            log_dict = {
                "group_size": group_size,
                "trained_prompt_num": trained_prompt_num,
                "zero_std_ratio": zero_std_ratio,
            }
            stat_tracker.clear()
        else:
            advantages = _normalize_rewards(gathered_rewards["avg"])

    return advantages, log_dict
