import json
import os
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Sampler


class TextPromptDataset(Dataset):
    def __init__(self, dataset: str, split: str = "train"):
        """Load plain text prompts for train/test splits.

        Args:
            dataset: Root dataset directory.
            split: File prefix (e.g., `train` or `test`).
        """
        self.file_path = os.path.join(dataset, f"{split}.txt")
        with open(self.file_path, "r") as f:
            self.prompts = [line.strip() for line in f.readlines()]

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int | Tuple[int, int]) -> Dict:
        """Return a prompt item, carrying sampler epoch_tag if provided."""
        epoch_tag = None
        if isinstance(idx, tuple):
            epoch_tag, idx = idx
        return {"epoch": epoch_tag, "prompt": self.prompts[idx], "metadata": {}}

    @staticmethod
    def collate_fn(examples: List[Dict]) -> Tuple[Optional[int], List[str], List[Dict]]:
        """Batch prompts while preserving a consistent epoch tag."""
        epoch_tags = [example.get("epoch") for example in examples]
        epoch_tag = (
            epoch_tags[0] if all(tag == epoch_tags[0] for tag in epoch_tags) else None
        )
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return epoch_tag, prompts, metadatas


class GenevalPromptDataset(Dataset):
    def __init__(self, dataset: str, split: str = "train"):
        """Load Geneval prompts with metadata for the given split."""
        self.file_path = os.path.join(dataset, f"{split}_metadata.jsonl")
        with open(self.file_path, "r", encoding="utf-8") as f:
            self.metadatas = [json.loads(line) for line in f]
            self.prompts = [item["prompt"] for item in self.metadatas]

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int | Tuple[int, int]) -> Dict:
        """Return a prompt+metadata, carrying sampler epoch_tag if provided."""
        epoch_tag = None
        if isinstance(idx, tuple):
            epoch_tag, idx = idx
        return {
            "epoch": epoch_tag,
            "prompt": self.prompts[idx],
            "metadata": self.metadatas[idx],
        }

    @staticmethod
    def collate_fn(examples: List[Dict]) -> Tuple[Optional[int], List[str], List[Dict]]:
        """Batch Geneval items while preserving epoch tags."""
        epoch_tags = [example.get("epoch") for example in examples]
        epoch_tag = (
            epoch_tags[0] if all(tag == epoch_tags[0] for tag in epoch_tags) else None
        )
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return epoch_tag, prompts, metadatas


class DistributedKRepeatSampler(Sampler):
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        k: int,
        num_replicas: int,
        rank: int,
        seed: int = 0,
    ):
        """Repeat each prompt k times per global batch and shard across ranks.

        Args:
            dataset: Dataset to sample from.
            batch_size: Per-rank batch size.
            k: Repetition factor per prompt.
            num_replicas: World size.
            rank: Current rank id.
            seed: Base seed for deterministic shuffles.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.k = k
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.total_samples = self.num_replicas * self.batch_size
        assert (
            self.total_samples % self.k == 0
        ), f"k can not div n*b, k{k}-num_replicas{num_replicas}-batch_size{batch_size}"
        self.m = self.total_samples // self.k
        self.epoch = 0

    def __iter__(self):
        """Yield per-rank batches with (epoch_tag, idx) pairs."""
        while True:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g)[: self.m].tolist()
            repeated_indices = [idx for idx in indices for _ in range(self.k)]
            shuffled_indices = torch.randperm(
                len(repeated_indices), generator=g
            ).tolist()
            shuffled_samples = [repeated_indices[i] for i in shuffled_indices]
            per_card_samples = []
            for i in range(self.num_replicas):
                start = i * self.batch_size
                end = start + self.batch_size
                per_card_samples.append(
                    [(self.epoch, idx) for idx in shuffled_samples[start:end]]
                )
            yield per_card_samples[self.rank]

    def set_epoch(self, epoch: int):
        """Set epoch tag to keep RNG in sync across workers."""
        self.epoch = epoch


def build_dataloaders(
    cfg, accelerator
) -> Tuple[DataLoader, DataLoader, DistributedKRepeatSampler]:
    """Construct train/eval dataloaders and sampler with epoch tags.

    Args:
        cfg: Parsed training configuration.
        accelerator: Accelerator instance to read rank/world info.

    Returns:
        Tuple of (train_dataloader, test_dataloader, train_sampler).
    """
    if cfg.prompt_fn == "general_ocr":
        train_dataset = TextPromptDataset(cfg.paths.dataset, "train")
        test_dataset = TextPromptDataset(cfg.paths.dataset, "test")
        collate_fn = TextPromptDataset.collate_fn
    elif cfg.prompt_fn == "geneval":
        train_dataset = GenevalPromptDataset(cfg.paths.dataset, "train")
        test_dataset = GenevalPromptDataset(cfg.paths.dataset, "test")
        collate_fn = GenevalPromptDataset.collate_fn
    else:
        raise NotImplementedError("Only general_ocr or geneval prompt_fn supported")

    train_sampler = DistributedKRepeatSampler(
        dataset=train_dataset,
        batch_size=cfg.sample.batch_size,
        k=cfg.sample.num_video_per_prompt,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        seed=42,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=1,
        collate_fn=collate_fn,
        prefetch_factor=1,
        persistent_workers=False,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=cfg.sample.eval_batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        num_workers=8,
    )
    return train_dataloader, test_dataloader, train_sampler
