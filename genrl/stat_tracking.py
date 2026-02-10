import numpy as np
from collections import defaultdict
import torch

from genrl.constants import EPSILON


class PerPromptStatTracker:
    def __init__(self, use_global_std: bool = False, max_group_std: bool = False):
        """Track per-prompt reward history and compute advantages.

        Args:
            use_global_std: If True, use global std across all rewards.
            max_group_std: If True, use the maximum std among all prompt groups.
        """
        self.use_global_std = use_global_std
        self.max_group_std = max_group_std
        self.stats = {}
        self.history_prompts = set()

    def update(self, prompts, rewards, mode: str = "grpo"):
        """Update stats and compute advantages for the given batch.

        Args:
            prompts: Iterable of prompt strings.
            rewards: Array-like rewards aligned with prompts.
            mode: Advantage shaping mode: grpo|rwr|sft|dpo.

        Returns:
            Advantages array aligned with `prompts`.
        """
        prompts = np.array(prompts)
        rewards = np.array(rewards, dtype=np.float64)
        unique = np.unique(prompts)
        advantages = np.empty_like(rewards) * 0.0
        # First pass: update stats for all prompts
        for prompt in unique:
            prompt_rewards = rewards[prompts == prompt]
            if prompt not in self.stats:
                self.stats[prompt] = []
            self.stats[prompt].extend(prompt_rewards)
            self.history_prompts.add(hash(prompt))
            # Convert to numpy array for computation
            self.stats[prompt] = np.stack(self.stats[prompt])

        # Compute max_std if max_group_std is enabled
        max_std = None
        if self.max_group_std and len(unique) > 0:
            prompt_stds = []
            for prompt in unique:
                prompt_std = np.std(self.stats[prompt], axis=0, keepdims=True) + EPSILON
                prompt_stds.append(prompt_std)
            # Find the maximum std value across all prompts
            max_std_value = max(np.max(std) for std in prompt_stds)
            # Use the shape of the first std and fill with max_std_value
            max_std = np.full_like(prompt_stds[0], max_std_value)

        # Second pass: compute advantages for each prompt
        for prompt in unique:
            prompt_rewards = rewards[prompts == prompt]
            mean = np.mean(self.stats[prompt], axis=0, keepdims=True)
            if self.use_global_std:
                std = np.std(rewards, axis=0, keepdims=True) + EPSILON
            elif self.max_group_std:
                std = max_std
            else:
                std = np.std(self.stats[prompt], axis=0, keepdims=True) + EPSILON
            if mode == "grpo":
                advantages[prompts == prompt] = (prompt_rewards - mean) / std
            elif mode == "rwr":
                advantages[prompts == prompt] = prompt_rewards
            elif mode == "sft":
                advantages[prompts == prompt] = (
                    (
                        torch.tensor(prompt_rewards)
                        == torch.max(torch.tensor(prompt_rewards))
                    )
                    .float()
                    .numpy()
                )
            elif mode == "dpo":
                prompt_advantages = torch.tensor(prompt_rewards)
                max_idx = torch.argmax(prompt_advantages)
                min_idx = torch.argmin(prompt_advantages)
                if max_idx == min_idx:
                    min_idx = 0
                    max_idx = 1
                result = torch.zeros_like(prompt_advantages).float()
                result[max_idx] = 1.0
                result[min_idx] = -1.0
                advantages[prompts == prompt] = result.numpy()
        return advantages

    def get_stats(self):
        """Return average group size and number of unique prompts seen.

        Returns:
            Tuple of (avg_group_size, num_unique_prompts_seen).
        """
        avg_group_size = (
            sum(len(v) for v in self.stats.values()) / len(self.stats)
            if self.stats
            else 0
        )
        history_prompts = len(self.history_prompts)
        return avg_group_size, history_prompts

    def clear(self):
        """Clear stored statistics."""
        self.stats = {}
